import copy
import gc
import logging
from typing import Callable, Optional, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xvec
from joblib import Parallel, delayed
from openeo_pg_parser_networkx.pg_schema import TemporalInterval, TemporalIntervals

from openeo_processes_dask_slim.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    TooManyDimensions,
)

from openeo_processes_dask_slim.process_implementations.cubes.dggs import (
    get_dggs_dim,
    is_dggs_cube,
)

__all__ = ["aggregate_temporal", "aggregate_temporal_period", "aggregate_spatial"]

logger = logging.getLogger(__name__)


def aggregate_temporal(
    data: RasterCube,
    intervals: Union[TemporalIntervals, list[TemporalInterval], list[Optional[str]]],
    reducer: Callable,
    labels: Optional[list] = None,
    dimension: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        t = dimension
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        t = temporal_dims[0]
    if isinstance(intervals, TemporalIntervals) or isinstance(intervals, list):
        interval_str = []
        for interval in intervals:
            if isinstance(interval, TemporalInterval):
                interval_0 = str(interval[0].root)
                interval_1 = str(interval[1].root)
                interval_str.append([interval_0, interval_1])
        if interval_str:
            intervals = interval_str

    intervals_np = (
        np.array(intervals, dtype=np.datetime64).astype("datetime64[s]").astype(float)
    )
    intervals_flat = np.reshape(
        intervals_np, np.shape(intervals_np)[0] * np.shape(intervals_np)[1]
    )

    if not labels:
        labels = np.array(intervals, dtype="datetime64[s]").astype(str)[:, 0]
    if (intervals_np[1:, 0] < intervals_np[:-1, 1]).any():
        raise NotImplementedError(
            "Aggregating data for overlapping time ranges is not implemented. "
        )

    mask = np.zeros((len(labels) * 2) - 2).astype(bool)
    mask[1::2] = np.isin(intervals_np[1:, 0], intervals_np[:-1, 1])
    mask = np.append(mask, np.array([False, True]))

    labels_nans = np.arange(len(labels) * 2).astype(str)
    labels_nans[::2] = labels
    labels_nans = labels_nans[~mask]

    intervals_flat = np.unique(intervals_flat)
    data_copy = copy.deepcopy(data)
    t_coords = data_copy[t].values.astype(str)
    data_copy = data_copy.assign_coords(
        {t: np.array(t_coords, dtype="datetime64[s]").astype(float)}
    )
    grouped_data = data_copy.groupby_bins(t, bins=intervals_flat)
    positional_parameters = {"data": 0}
    groups = grouped_data.reduce(
        reducer, keep_attrs=True, positional_parameters=positional_parameters
    )
    groups = groups.assign_coords({t + "_bins": labels_nans})
    data_agg_temp = groups.sel({t + "_bins": labels})
    data_agg_temp = data_agg_temp.rename({t + "_bins": t})

    return data_agg_temp


def get_intervals(data, period):
    format = "%Y-%m-%dT%H:%M:%S"
    start, end = data["t"].values[0], data["t"].values[-1]
    year_start = pd.to_datetime(start).year
    year_end = pd.to_datetime(end).year
    month_start = pd.to_datetime(start).month
    month_end = pd.to_datetime(end).month

    if period == "decade":
        year_start = np.datetime64(
            (np.floor(year_start / 10) * 10).astype(int).astype(str)
        )
        year_end = np.datetime64((np.ceil(year_end / 10) * 10).astype(int).astype(str))
        intervals = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            format
        )
        labels = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            "%Y"
        )[:-1]
    elif period == "decade-ad":
        year_start = np.datetime64(
            (np.floor(year_start / 10) * 10 + 1).astype(int).astype(str)
        )
        year_end = np.datetime64(
            (np.ceil(year_end / 10) * 10 + 1).astype(int).astype(str)
        )
        intervals = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            format
        )
        labels = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            "%Y"
        )[:-1]
    elif period == "tropical-season":
        if month_start >= 5 and month_start < 10:
            month_start = np.datetime64(str(year_start) + "-05-01")
        elif month_start < 5:
            month_start = np.datetime64(str(year_start - 1) + "-11-01")
        else:
            month_start = np.datetime64(str(year_start) + "-11-01")
        if month_end >= 5 and month_end < 10:
            month_end = np.datetime64(str(year_end) + "-11-01")
        elif month_end < 5:
            month_end = np.datetime64(str(year_end) + "-05-01")
        else:
            month_end = np.datetime64(str(year_end + 1) + "-05-01")
        intervals = pd.period_range(
            start=month_start, end=month_end, freq="6M"
        ).strftime(format)
        labels = []
        for interval in intervals[:-1]:
            if "-11-" in interval:
                labels.append(interval[:5] + "ndjfma")
            if "-05-" in interval:
                labels.append(interval[:5] + "mjjaso")
    elif period == "dekad":
        day = pd.to_datetime(start).day
        day_start = (np.floor(day / 10) * 10 + 1).astype(int).astype(str)
        day_start = f"{year_start}-{month_start}-{day_start}"
        intervals = pd.date_range(
            start=day_start, end=f"{year_start}-{month_start}-22", freq="10D"
        ).strftime(format)
        for date in pd.date_range(
            start=f"{year_start}-{month_start}-22", end=end, freq="1MS"
        )[:-1]:
            intervals = intervals.append(
                pd.date_range(start=date, freq="10D", periods=3).strftime(format)
            )
        day = pd.to_datetime(end).day
        periods = (np.ceil((day - 1) / 10)).astype(int)
        day_end = f"{year_end}-{month_end}-01"
        if day > 21:
            day_end = pd.date_range(start=day_end, freq="10D", periods=periods)
            days = 7
            last_day = day_end[-1] + pd.DateOffset(days=7)
            while last_day.day != 1:
                days += 1
                last_day = day_end[-1] + pd.DateOffset(days=days)
            day_end = day_end.append(pd.DatetimeIndex(data=[str(last_day)])).strftime(
                format
            )
        else:
            day_end = pd.date_range(
                start=day_end, freq="10D", periods=periods + 1
            ).strftime(format)
        intervals = intervals.append(day_end)
        labels = []
        for interval in intervals[:-1]:
            year = pd.DatetimeIndex(data=[interval]).year.astype(int)[0]
            dekad = int(pd.DatetimeIndex(data=[interval]).day_of_year[0] / 10)
            label = f"{year}-{dekad}" if dekad > 9 else f"{year}-0{dekad}"
            labels.append(label)
    else:
        raise NotImplementedError(
            f"The provided period '{period})' is not implemented. "
        )
    interval_array = np.array(intervals, dtype=str)
    interval_matrix = np.zeros((len(interval_array) - 1, 2)).astype(str)
    interval_matrix[:, 0] = interval_array[:-1]
    interval_matrix[:, 1] = interval_array[1:]
    return interval_matrix, list(labels)


def aggregate_temporal_period(
    data: RasterCube,
    reducer: Callable,
    period: str,
    dimension: Optional[str] = None,
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
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

    periods_to_frequency = {
        "hour": "H",
        "day": "D",
        "week": "W",
        "month": "M",
        "season": "QS-DEC",
        "year": "AS",
    }

    if period in periods_to_frequency.keys():
        frequency = periods_to_frequency[period]
        resampled_data = data.resample({applicable_temporal_dimension: frequency})

        positional_parameters = {"data": 0}
        return resampled_data.reduce(
            reducer, keep_attrs=True, positional_parameters=positional_parameters
        )

    else:
        intervals, labels = get_intervals(data, period)
        return aggregate_temporal(
            data=data, intervals=intervals, reducer=reducer, labels=labels
        )


def _load_geometries_for_aggregate(geometries):
    import geopandas as gpd

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


def _aggregate_spatial_dggs(
    data: RasterCube,
    geometries: dict,
    reducer: Callable,
    context: Optional[dict] = None,
) -> RasterCube:
    dggs_dim = get_dggs_dim(data)

    if "lon" not in data.coords or "lat" not in data.coords:
        raise DimensionNotAvailable(
            "DGGS cube must have 'lat' and 'lon' coordinates for aggregate_spatial."
        )

    gdf = _load_geometries_for_aggregate(geometries)
    lon = np.asarray(data["lon"].data)
    lat = np.asarray(data["lat"].data)

    result_bands = {}
    for geom_idx, (_, row) in enumerate(gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        bounds = geom.bounds
        bbox_mask = (
            (lon >= bounds[0]) & (lon <= bounds[2])
            & (lat >= bounds[1]) & (lat <= bounds[3])
        )
        bbox_indices = np.where(bbox_mask)[0]

        if len(bbox_indices) == 0:
            continue

        contained = np.array([
            geom.contains(shapely.geometry.Point(lon[i], lat[i]))
            for i in bbox_indices
        ])
        cell_indices = bbox_indices[contained]

        if len(cell_indices) == 0:
            continue

        cell_data = data.isel(**{dggs_dim: cell_indices})
        positional_parameters = {"data": 0}
        aggregated = cell_data.reduce(
            reducer,
            dim=dggs_dim,
            keep_attrs=False,
            positional_parameters=positional_parameters,
            context=context,
        )

        for var_name in aggregated.data_vars:
            if var_name not in result_bands:
                result_bands[var_name] = []
            result_bands[var_name].append(aggregated[var_name])

    if not result_bands:
        from openeo_processes_dask_slim.process_implementations.exceptions import (
            NoDataAvailable,
        )

        raise NoDataAvailable(
            "No DGGS cells intersect any of the specified geometries."
        )

    out_vars = {}
    for var_name, arrays in result_bands.items():
        if len(arrays) == 1:
            stacked = arrays[0].expand_dims("geometry")
        else:
            stacked = xr.concat(arrays, dim="geometry")
        out_vars[var_name] = stacked

    n_geom = len(next(iter(result_bands.values())))
    result = xr.Dataset(
        data_vars=out_vars,
        coords={"geometry": np.arange(n_geom)},
        attrs=data.attrs,
    )
    return result


def aggregate_spatial(
    data: RasterCube,
    geometries: dict,
    reducer: Callable,
    context: Optional[dict] = None,
) -> RasterCube:
    if is_dggs_cube(data):
        return _aggregate_spatial_dggs(data, geometries, reducer, context)

    raise NotImplementedError(
        f"aggregate_spatial for planar x/y cubes is not yet implemented."
    )
