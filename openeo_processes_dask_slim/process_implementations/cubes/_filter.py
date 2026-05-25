import json
import logging
import warnings
from typing import Any, Callable, Optional

import dask.array as da
import geopandas as gpd
import numpy as np
import pyproj
import shapely
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

from openeo_processes_dask_slim.process_implementations.cubes.dggs import (
    get_dggs_dim,
    is_dggs_cube,
)
from openeo_processes_dask_slim.process_implementations.data_model import RasterCube
from openeo_processes_dask_slim.process_implementations.exceptions import (
    BandFilterParameterMissing,
    DimensionMissing,
    DimensionNotAvailable,
    NoDataAvailable,
    TemporalExtentEmpty,
    TooManyDimensions,
)

DEFAULT_CRS = "EPSG:4326"


logger = logging.getLogger(__name__)

__all__ = [
    "filter_labels",
    "filter_temporal",
    "filter_bands",
    "filter_bbox",
    "filter_spatial",
]


def filter_temporal(
    data: RasterCube, extent: TemporalInterval, dimension: str = None
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
        if dimension not in temporal_dims:
            logger.warning(
                f"The selected dimension {dimension} exists but it is not labeled as a temporal dimension. Available temporal diemnsions are {temporal_dims}."
            )
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        applicable_temporal_dimension = temporal_dims[0]

    # This line raises a deprecation warning, which according to this thread
    # will never actually be deprecated:
    # https://github.com/numpy/numpy/issues/23904
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if isinstance(extent, TemporalInterval):
            start_time = extent.start
            end_time = extent.end
        else:
            start_time = extent[0]
            end_time = extent[1]

        if isinstance(start_time, str):
            start_time = np.datetime64(start_time)
        elif start_time is not None:
            start_time = start_time.to_numpy()

        if isinstance(end_time, str):
            end_time = np.datetime64(end_time)
        elif end_time is not None:
            end_time = end_time.to_numpy()

        # The second element is the end of the temporal interval.
        # The specified instance in time is excluded from the interval.
        # See https://processes.openeo.org/#filter_temporal
        if end_time is not None:
            end_time -= np.timedelta64(1, "ms")

        if start_time is not None and end_time is not None and end_time < start_time:
            raise TemporalExtentEmpty(
                "The temporal extent is empty. The second instant in time must always be greater/later than the first instant in time."
            )

        data = data.where(~np.isnat(data[applicable_temporal_dimension]), drop=True)
        filtered = data.loc[
            {applicable_temporal_dimension: slice(start_time, end_time)}
        ]

    return filtered


def filter_labels(
    data: RasterCube, condition: Callable, dimension: str, context: Optional[Any] = None
) -> RasterCube:
    if isinstance(data, xr.Dataset) and dimension == "bands":
        labels = list(data.data_vars)
        if not context:
            context = {}
        positional_parameters = {"x": 0}
        named_parameters = {"x": labels, "context": context}
        selected = [
            name
            for name in labels
            if condition(
                name,
                positional_parameters=positional_parameters,
                named_parameters=named_parameters,
            )
        ]
        return data[selected]

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    labels = np.array(data[dimension].values)
    if not context:
        context = {}
    positional_parameters = {"x": 0, "value": 0}
    named_parameters = {"x": labels, "value": labels, "context": context}
    filter_condition = np.vectorize(condition)
    filtered_labels = filter_condition(
        labels,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )
    label = np.argwhere(filtered_labels)
    data = data.isel(**{dimension: label[0]})
    return data


def filter_bands(data: RasterCube, bands: list[str] = None) -> RasterCube:
    if bands is None:
        raise BandFilterParameterMissing(
            "The process `filter_bands` requires the parameters `bands` to be set."
        )

    if isinstance(data, xr.Dataset):
        missing = [b for b in bands if b not in data.data_vars]
        if missing:
            raise Exception(
                f"The provided bands: {bands} are not all available in the datacube. Please modify the bands parameter of filter_bands and choose among: {list(data.data_vars)}."
            )
        return data[bands]

    if len(data.openeo.band_dims) < 1:
        raise DimensionMissing("A band dimension is missing.")
    band_dim = data.openeo.band_dims[0]

    try:
        data = data.sel(**{band_dim: bands})
    except Exception as e:
        raise Exception(
            f"The provided bands: {bands} are not all available in the datacube. Please modify the bands parameter of filter_bands and choose among: {data[band_dim].values}."
        )
    return data


def filter_bbox(data: RasterCube, extent: BoundingBox) -> RasterCube:
    y_dim = data.openeo.y_dim
    x_dim = data.openeo.x_dim

    if y_dim is None and x_dim is None:
        if is_dggs_cube(data):
            return _filter_bbox_dggs(data, extent)
        raise DimensionNotAvailable(
            "No spatial dimensions available, can't apply filter_bbox."
        )

    try:
        odc_crs = data.odc.crs
        if odc_crs is not None:
            input_crs = str(odc_crs)
        else:
            input_crs = data.attrs.get("crs", None)
    except Exception as e:
        raise Exception(f"Not possible to estimate the input data projection! {e}")
    if input_crs is not None and not pyproj.crs.CRS(extent.crs).equals(input_crs):
        reprojected_extent = _reproject_bbox(extent, input_crs)
    else:
        reprojected_extent = extent

    if y_dim is not None:
        # Check if the coordinates are increasing or decreasing
        if len(data[y_dim]) > 1:
            if data[y_dim][0] > data[y_dim][1]:
                y_slice = slice(reprojected_extent.north, reprojected_extent.south)
            else:
                y_slice = slice(reprojected_extent.south, reprojected_extent.north)
        else:
            # We need to check if the bbox crosses this single coordinate
            # if data[y_dim][0] < reprojected_extent.north and data[y_dim][0] > reprojected_extent.south:
            #     # bbox crosses the single coordinate
            #     y_slice = data[y_dim][0]
            # else:
            #     # bbox doesn't cross the single coordinate: return empty data or error?
            raise NotImplementedError(
                f"filter_bbox can't filter data with a single coordinate on {y_dim} yet."
            )

    if x_dim is not None:
        if len(data[x_dim]) > 1:
            if data[x_dim][0] > data[x_dim][1]:
                x_slice = slice(reprojected_extent.east, reprojected_extent.west)
            else:
                x_slice = slice(reprojected_extent.west, reprojected_extent.east)
        else:
            # We need to check if the bbox crosses this single coordinate. How to do this correctly?
            # if data[x_dim][0] < reprojected_extent.east and data[x_dim][0] > reprojected_extent.west:
            #     # bbox crosses the single coordinate
            #     y_slice = data[x_dim][0]
            # else:
            #     # bbox doesn't cross the single coordinate: return empty data or error?
            raise NotImplementedError(
                f"filter_bbox can't filter data with a single coordinate on {x_dim} yet."
            )

    if y_dim is not None and x_dim is not None:
        aoi = data.loc[{y_dim: y_slice, x_dim: x_slice}]
    elif x_dim is None:
        aoi = data.loc[{y_dim: y_slice}]
    else:
        aoi = data.loc[{x_dim: x_slice}]

    return aoi


def _filter_bbox_dggs(data: RasterCube, extent: BoundingBox) -> RasterCube:
    dggs_dim = get_dggs_dim(data)

    if "lon" not in data.coords or "lat" not in data.coords:
        raise DimensionNotAvailable(
            "DGGS cube must have 'lat' and 'lon' coordinates over the cell dimension "
            "for filter_bbox spatial filtering."
        )

    lon = np.asarray(data["lon"].data)
    lat = np.asarray(data["lat"].data)

    if extent.crs is not None and not pyproj.crs.CRS(extent.crs).equals("EPSG:4326"):
        transformer = pyproj.Transformer.from_crs(
            extent.crs, "EPSG:4326", always_xy=True
        )
        sw_lon, sw_lat = transformer.transform(extent.west, extent.south)
        ne_lon, ne_lat = transformer.transform(extent.east, extent.north)
        west, south = min(sw_lon, ne_lon), min(sw_lat, ne_lat)
        east, north = max(sw_lon, ne_lon), max(sw_lat, ne_lat)
    else:
        west, south = extent.west, extent.south
        east, north = extent.east, extent.north

    if west <= east:
        lon_mask = (lon >= west) & (lon <= east)
    else:
        lon_mask = (lon >= west) | (lon <= east)

    lat_mask = (lat >= south) & (lat <= north)
    mask = lon_mask & lat_mask

    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise NoDataAvailable("No DGGS cells intersect the specified bounding box.")

    return data.isel(**{dggs_dim: indices})


def _reproject_bbox(extent: BoundingBox, target_crs: str) -> BoundingBox:
    bbox_points = [
        [extent.south, extent.west],
        [extent.south, extent.east],
        [extent.north, extent.east],
        [extent.north, extent.west],
    ]
    if extent.crs is not None:
        source_crs = extent.crs
    else:
        source_crs = "EPSG:4326"

    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    x_reprojected = []
    y_reprojected = []
    for p in bbox_points:
        x1, y1 = p
        x2, y2 = transformer.transform(y1, x1)
        x_reprojected.append(x2)
        y_reprojected.append(y2)

    x_reprojected = np.array(x_reprojected)
    y_reprojected = np.array(y_reprojected)

    reprojected_extent = {}

    reprojected_extent = BoundingBox(
        west=x_reprojected.min(),
        east=x_reprojected.max(),
        north=y_reprojected.max(),
        south=y_reprojected.min(),
        crs=target_crs,
    )
    return reprojected_extent


def filter_spatial(data: RasterCube, geometries: dict) -> RasterCube:
    if is_dggs_cube(data):
        return _filter_spatial_dggs(data, geometries)

    y_dim = data.openeo.y_dim
    x_dim = data.openeo.x_dim

    if y_dim is None or x_dim is None:
        raise DimensionNotAvailable(
            "Planar spatial dimensions (x, y) not found on data cube."
        )

    if isinstance(geometries, dict):
        gdf = gpd.GeoDataFrame.from_features(geometries, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(geometries, crs="EPSG:4326")

    if gdf.crs is not None:
        try:
            odc_crs = data.odc.crs
        except Exception:
            odc_crs = data.attrs.get("crs", None)
        if odc_crs is not None and not pyproj.crs.CRS(str(gdf.crs)).equals(odc_crs):
            gdf = gdf.to_crs(odc_crs)

    union_geom = gdf.geometry.unary_union
    if union_geom is None or union_geom.is_empty:
        raise NoDataAvailable("Empty geometry provided to filter_spatial.")

    bounds = union_geom.bounds
    x_coords = np.asarray(data[x_dim].data)
    y_coords = np.asarray(data[y_dim].data)

    x_mask = (x_coords >= bounds[0]) & (x_coords <= bounds[2])
    y_mask = (y_coords >= bounds[1]) & (y_coords <= bounds[3])
    bbox_sel = data.isel(**{x_dim: np.where(x_mask)[0], y_dim: np.where(y_mask)[0]})

    xx, yy = np.meshgrid(
        np.asarray(bbox_sel[x_dim].data),
        np.asarray(bbox_sel[y_dim].data),
    )
    points = np.column_stack([xx.ravel(), yy.ravel()])
    point_geoms = [shapely.geometry.Point(p[0], p[1]) for p in points]
    inside = np.array([union_geom.contains(p) for p in point_geoms]).reshape(xx.shape)

    if not inside.any():
        raise NoDataAvailable("No pixels intersect the specified geometries.")

    result = bbox_sel.where(xr.DataArray(inside, dims=(y_dim, x_dim)), drop=True)

    return result


def _load_geometries_dggs(geometries: dict):
    if isinstance(geometries, dict):
        geom_type = geometries.get("type", None)
        if geom_type == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geometries, crs="EPSG:4326")
        elif geom_type == "Feature":
            gdf = gpd.GeoDataFrame.from_features(
                {"type": "FeatureCollection", "features": [geometries]},
                crs="EPSG:4326",
            )
        elif geom_type in ("Polygon", "MultiPolygon"):
            gdf = gpd.GeoDataFrame(
                geometry=[shapely.geometry.shape(geometries)], crs="EPSG:4326"
            )
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geom_type}")
    else:
        raise ValueError("Geometries must be a GeoJSON dict.")
    return gdf


def _filter_spatial_dggs(data: RasterCube, geometries: dict) -> RasterCube:
    dggs_dim = get_dggs_dim(data)

    if "lon" not in data.coords or "lat" not in data.coords:
        raise DimensionNotAvailable(
            "DGGS cube must have 'lat' and 'lon' coordinates " "for filter_spatial."
        )

    gdf = _load_geometries_dggs(geometries)

    if gdf.crs is not None and not pyproj.crs.CRS(str(gdf.crs)).equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")

    union_geom = gdf.geometry.unary_union
    if union_geom is None or union_geom.is_empty:
        raise NoDataAvailable("Empty geometry provided to filter_spatial.")

    lon = np.asarray(data["lon"].data)
    lat = np.asarray(data["lat"].data)

    bounds = union_geom.bounds
    bbox_mask = (
        (lon >= bounds[0])
        & (lon <= bounds[2])
        & (lat >= bounds[1])
        & (lat <= bounds[3])
    )
    bbox_indices = np.where(bbox_mask)[0]

    if len(bbox_indices) == 0:
        raise NoDataAvailable("No DGGS cells intersect the specified geometries.")

    contained = np.array(
        [
            union_geom.contains(shapely.geometry.Point(lon[i], lat[i]))
            for i in bbox_indices
        ]
    )
    final_indices = bbox_indices[contained]

    if len(final_indices) == 0:
        raise NoDataAvailable("No DGGS cells intersect the specified geometries.")

    return data.isel(**{dggs_dim: final_indices})
