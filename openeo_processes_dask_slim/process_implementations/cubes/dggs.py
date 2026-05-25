import logging
from typing import Callable, Optional

import numpy as np
import pyproj
import shapely
import xarray as xr

from openeo_processes_dask_slim.process_implementations.data_model import RasterCube
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    OpenEOException,
)

try:
    import astropy_healpix as ah
    from astropy_healpix.high_level import neighbours, nested_to_ring, ring_to_nested

    HAS_ASTROPY_HEALPIX = True
except ImportError:
    HAS_ASTROPY_HEALPIX = False

logger = logging.getLogger(__name__)

DGGS_DIM_ATTR = "dggs_cell_id_dim"
DGGS_SYSTEM_ATTR = "dggs_grid_system"
DGGS_RES_ATTR = "dggs_resolution"
DGGS_ORDER_ATTR = "dggs_order"

HEALPIX_DIM_GUESSES = ["healpix_index"]

__all__ = [
    "is_dggs_cube",
    "get_dggs_dim",
    "get_dggs_grid_system",
    "get_dggs_resolution",
    "get_dggs_order",
    "get_dggs_crs",
    "require_dggs_cube",
    "is_healpix_cube",
    "filter_dggs",
    "mask_polygon_dggs",
    "resample_dggs",
    "dggs_to_raster",
    "apply_neighborhood_dggs",
    "mask_polygon_dggs",
]


def _has_dggs_attrs(data: RasterCube) -> bool:
    return DGGS_SYSTEM_ATTR in data.attrs and DGGS_DIM_ATTR in data.attrs


def _has_healpix_attrs(data: RasterCube) -> bool:
    crs = data.attrs.get("crs", "")
    return "healpix_nside" in data.attrs or str(crs).lower().startswith("healpix:")


def _detect_dggs_dim_from_attrs(data: RasterCube) -> Optional[str]:
    dim_name = data.attrs.get(DGGS_DIM_ATTR, None)
    if dim_name is not None and dim_name in data.dims:
        return dim_name
    return None


def _detect_healpix_dim_from_name(data: RasterCube) -> Optional[str]:
    for dim in HEALPIX_DIM_GUESSES:
        if dim in data.dims:
            return dim
    return None


def is_dggs_cube(data: RasterCube) -> bool:
    if _has_dggs_attrs(data):
        return _detect_dggs_dim_from_attrs(data) is not None
    if _has_healpix_attrs(data):
        return _detect_healpix_dim_from_name(data) is not None
    return bool(_detect_healpix_dim_from_name(data) is not None)


def is_healpix_cube(data: RasterCube) -> bool:
    return (
        get_dggs_grid_system(data) == "healpix"
        or _has_healpix_attrs(data)
        or _detect_healpix_dim_from_name(data) is not None
    )


def get_dggs_dim(data: RasterCube) -> Optional[str]:
    dim = _detect_dggs_dim_from_attrs(data)
    if dim is not None:
        return dim
    if is_healpix_cube(data):
        return _detect_healpix_dim_from_name(data)
    return None


def get_dggs_grid_system(data: RasterCube) -> Optional[str]:
    return data.attrs.get(DGGS_SYSTEM_ATTR, None)


def get_dggs_resolution(data: RasterCube) -> Optional[int]:
    res = data.attrs.get(DGGS_RES_ATTR, None)
    if res is not None:
        return int(res)
    return data.attrs.get("healpix_nside", None)


def get_dggs_order(data: RasterCube) -> Optional[str]:
    return data.attrs.get(DGGS_ORDER_ATTR, data.attrs.get("healpix_order", None))


def get_dggs_crs(data: RasterCube) -> Optional[str]:
    return data.attrs.get("crs", None)


def require_dggs_cube(data: RasterCube) -> str:
    if not is_dggs_cube(data):
        raise DimensionNotAvailable(
            "The input data cube is not a DGGS cube. "
            "A DGGS cube must have a cell ID dimension "
            "and DGGS metadata attributes (dggs_grid_system, dggs_cell_id_dim)."
        )
    dggs_dim = get_dggs_dim(data)
    if dggs_dim is None:
        raise DimensionNotAvailable("No DGGS cell ID dimension found in the data cube.")
    return dggs_dim


def filter_dggs(
    data: RasterCube,
    cells: Optional[list[int]] = None,
    parent_cells: Optional[list[int]] = None,
    nside: Optional[int] = None,
    order: Optional[str] = None,
) -> RasterCube:
    dggs_dim = require_dggs_cube(data)

    if cells is None and parent_cells is None:
        raise ValueError("At least one of `cells` or `parent_cells` must be specified.")

    selected = set()

    if cells is not None:
        selected.update(int(c) for c in cells)

    if parent_cells is not None:
        if not HAS_ASTROPY_HEALPIX:
            raise ImportError(
                "astropy-healpix is required for parent cell selection. "
                "Install it with: pip install astropy-healpix"
            )

        cube_nside = get_dggs_resolution(data)
        if cube_nside is None:
            raise DimensionNotAvailable(
                "Cannot determine DGGS resolution for parent cell selection."
            )

        parent_nside = nside if nside is not None else cube_nside
        if parent_nside > cube_nside:
            raise ValueError(
                f"Parent nside ({parent_nside}) must be <= cube nside ({cube_nside})."
            )

        use_order = order if order is not None else (get_dggs_order(data) or "nested")
        if use_order == "ring":
            logger.warning(
                "Parent cell selection with ring ordering may not produce "
                "the correct hierarchical children. Use nested ordering "
                "for reliable parent-child queries."
            )

        level_diff = int(np.log2(cube_nside // parent_nside))
        scale = 4**level_diff

        for parent_pix in parent_cells:
            start = int(parent_pix) * scale
            for offset in range(scale):
                selected.add(start + offset)

    sorted_selected = sorted(selected)
    return data.isel(**{dggs_dim: sorted_selected})


def _healpix_parent_map(
    fine_nside: int, coarse_nside: int, order: str = "nested"
) -> np.ndarray:
    if coarse_nside > fine_nside:
        raise ValueError(
            f"Coarse nside ({coarse_nside}) must be <= fine nside ({fine_nside})."
        )
    scale = (fine_nside // coarse_nside) ** 2
    npix = ah.nside_to_npix(fine_nside)
    fine_indices = np.arange(npix)

    if order == "nested":
        coarse_indices = fine_indices // scale
    else:
        nested = ring_to_nested(fine_indices, fine_nside)
        coarse_nested = nested // scale
        coarse_indices = nested_to_ring(coarse_nested, coarse_nside)

    return coarse_indices


def resample_dggs(
    data: RasterCube,
    resolution: int,
    order: Optional[str] = None,
    reducer: str = "mean",
) -> RasterCube:
    dggs_dim = require_dggs_cube(data)

    if not is_healpix_cube(data):
        raise NotImplementedError(
            "resample_dggs is currently only implemented for HEALPix cubes."
        )
    if not HAS_ASTROPY_HEALPIX:
        raise ImportError(
            "astropy-healpix is required for resample_dggs. "
            "Install it with: pip install astropy-healpix"
        )

    cube_nside = get_dggs_resolution(data)
    if cube_nside is None:
        raise DimensionNotAvailable("Cannot determine cube DGGS resolution.")

    if resolution > cube_nside:
        raise NotImplementedError(
            f"Upsampling from nside={cube_nside} to nside={resolution} "
            f"is not supported. Only downsampling is implemented."
        )
    if resolution == cube_nside:
        return data

    use_order = order if order is not None else (get_dggs_order(data) or "nested")
    if use_order not in ("ring", "nested"):
        raise ValueError(f"Unsupported order: {use_order}")

    parent_map = _healpix_parent_map(cube_nside, resolution, order=use_order)
    data = data.assign_coords(**{f"{dggs_dim}_parent": (dggs_dim, parent_map)})

    positional_parameters = {"data": 0}
    reducer_map = {
        "mean": lambda x, **kw: np.nanmean(x, **kw),
        "sum": lambda x, **kw: np.nansum(x, **kw),
        "min": lambda x, **kw: np.nanmin(x, **kw),
        "max": lambda x, **kw: np.nanmax(x, **kw),
        "count": lambda x, **kw: np.sum(~np.isnan(x), **kw),
        "mode": lambda x, **kw: _mode_reducer(x, **kw),
    }
    if reducer not in reducer_map:
        raise ValueError(
            f"Unsupported reducer '{reducer}'. "
            f"Supported: {list(reducer_map.keys())}"
        )

    def _reducer_wrapper(data_array, **kwargs):
        result = data_array.reduce(
            reducer_map[reducer],
            dim=dggs_dim,
            keep_attrs=False,
            positional_parameters=positional_parameters,
        )
        return result

    result_arrays = {}
    for var_name in data.data_vars:
        grouped = data[var_name].groupby(f"{dggs_dim}_parent")
        reduced = grouped.reduce(reducer_map[reducer])
        result_arrays[var_name] = reduced

    result = xr.Dataset(result_arrays, attrs=data.attrs)

    coarse_npix = ah.nside_to_npix(resolution)
    coarse_indices = np.arange(coarse_npix)

    lon_rad, lat_rad = ah.healpix_to_lonlat(coarse_indices, resolution, order="ring")
    lat_vals = np.rad2deg(np.asarray(lat_rad))
    lon_vals = np.rad2deg(np.asarray(lon_rad))

    result = result.assign_coords(
        {
            dggs_dim: coarse_indices,
            "lat": (dggs_dim, lat_vals),
            "lon": (dggs_dim, lon_vals),
        }
    )
    result.attrs["crs"] = f"healpix:{resolution}"
    result.attrs["healpix_nside"] = resolution
    result.attrs["healpix_order"] = use_order

    return result


def _mode_reducer(data, axis=None, keepdims=False, **kwargs):
    from scipy import stats

    return stats.mode(data, axis=axis, keepdims=keepdims)[0]


def dggs_to_raster(
    data: RasterCube,
    projection: int = 4326,
    resolution: Optional[float] = None,
    bbox: Optional[dict] = None,
    method: str = "nearest",
) -> RasterCube:
    dggs_dim = require_dggs_cube(data)

    if not is_healpix_cube(data):
        raise NotImplementedError(
            "dggs_to_raster is currently only implemented for HEALPix cubes."
        )
    if not HAS_ASTROPY_HEALPIX:
        raise ImportError(
            "astropy-healpix is required for dggs_to_raster. "
            "Install it with: pip install astropy-healpix"
        )
    if method != "nearest":
        raise ValueError(f"Unsupported method '{method}'. Only 'nearest' is supported.")

    cube_nside = get_dggs_resolution(data)
    if cube_nside is None:
        raise DimensionNotAvailable("Cannot determine cube DGGS resolution.")

    if "lon" not in data.coords or "lat" not in data.coords:
        raise DimensionNotAvailable(
            "DGGS cube must have 'lat' and 'lon' coordinates for dggs_to_raster."
        )

    cube_lon = np.asarray(data["lon"].data)
    cube_lat = np.asarray(data["lat"].data)

    if bbox is not None:
        west = bbox.get("west", cube_lon.min())
        east = bbox.get("east", cube_lon.max())
        south = bbox.get("south", cube_lat.min())
        north = bbox.get("north", cube_lat.max())
    else:
        west, east = float(cube_lon.min()), float(cube_lon.max())
        south, north = float(cube_lat.min()), float(cube_lat.max())

    if resolution is None:
        resolution = (east - west) / max(360, int(np.sqrt(len(cube_lon))))
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")

    x_coords = np.arange(west, east, resolution)
    y_coords = np.arange(south, north, resolution)
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise OpenEOException("Target grid has zero extent.")

    xx, yy = np.meshgrid(x_coords, y_coords)
    flat_lon = xx.ravel()
    flat_lat = yy.ravel()

    from astropy import units as u

    flat_lon_q = np.deg2rad(flat_lon) * u.rad
    flat_lat_q = np.deg2rad(flat_lat) * u.rad
    target_pixels = ah.lonlat_to_healpix(
        flat_lon_q, flat_lat_q, cube_nside, order="ring"
    )

    flat_pixel_lookup = xr.DataArray(
        np.asarray(target_pixels).astype(int), dims="grid_index"
    )

    valid = (
        (flat_lon >= cube_lon.min())
        & (flat_lon <= cube_lon.max())
        & (flat_lat >= cube_lat.min())
        & (flat_lat <= cube_lat.max())
    )

    from astropy import units as u

    flat_lon_q = flat_lon * u.rad
    flat_lat_q = flat_lat * u.rad
    target_pixels = ah.lonlat_to_healpix(
        flat_lon_q, flat_lat_q, cube_nside, order="ring"
    )

    flat_pixel_lookup = xr.DataArray(
        np.asarray(target_pixels).astype(int), dims="grid_index"
    )

    result_vars = {}
    for var_name in data.data_vars:
        selected = data[var_name].isel(**{dggs_dim: flat_pixel_lookup})

        grid_y = xr.DataArray(np.repeat(y_coords, len(x_coords)), dims="grid_index")
        grid_x = xr.DataArray(np.tile(x_coords, len(y_coords)), dims="grid_index")
        selected = selected.assign_coords(grid_y=grid_y, grid_x=grid_x)
        selected = selected.set_index(grid_index=("grid_y", "grid_x"))
        unstacked = selected.unstack("grid_index")
        unstacked = unstacked.rename({"grid_y": "y", "grid_x": "x"})
        result_vars[var_name] = unstacked

    if not result_vars:
        raise OpenEOException("No data variables to convert.")

    result = xr.Dataset(
        result_vars,
        attrs=data.attrs,
    )
    result.attrs["crs"] = f"EPSG:{projection}"

    return result


def apply_neighborhood_dggs(
    data: RasterCube,
    process: Callable,
    radius: Optional[float] = None,
    k: Optional[int] = None,
    include_center: bool = True,
    context: Optional[dict] = None,
) -> RasterCube:
    dggs_dim = require_dggs_cube(data)

    if (k is None) == (radius is None):
        raise ValueError("Exactly one of `k` or `radius` must be specified.")
    if k is not None and (k < 1 or k > 4):
        raise ValueError("Only k in range 1-4 is supported in v1.")
    if radius is not None:
        raise NotImplementedError(
            "Angular radius neighborhoods are not yet implemented. Use `k` instead."
        )
    if not is_healpix_cube(data):
        raise NotImplementedError(
            "apply_neighborhood_dggs is currently only implemented for HEALPix cubes."
        )
    if not HAS_ASTROPY_HEALPIX:
        raise ImportError("astropy-healpix is required for apply_neighborhood_dggs.")

    nside = get_dggs_resolution(data)
    if nside is None:
        raise DimensionNotAvailable("Cannot determine cube DGGS resolution.")

    healpix_indices = np.asarray(data[dggs_dim].data)

    pixel_to_nbrs = {}
    max_nbrs = 0
    for pix in healpix_indices:
        pix_int = int(pix)
        collected = set()
        boundary = {pix_int}
        for _ in range(k):
            next_boundary = set()
            for p in boundary:
                nbrs = neighbours(np.array([p]), nside)
                nbrs = nbrs[nbrs >= 0]
                for nb in nbrs:
                    nb_int = int(nb)
                    if nb_int not in collected:
                        next_boundary.add(nb_int)
            collected.update(boundary)
            boundary = next_boundary
        collected.update(boundary)
        if not include_center:
            collected.discard(pix_int)
        sorted_nbrs = sorted(collected)
        pixel_to_nbrs[pix_int] = sorted_nbrs
        max_nbrs = max(max_nbrs, len(sorted_nbrs))

    n_pix = len(healpix_indices)
    neighbor_matrix = np.full((n_pix, max_nbrs), -1, dtype=np.int64)
    for i, pix in enumerate(healpix_indices):
        nbrs = pixel_to_nbrs[int(pix)]
        neighbor_matrix[i, : len(nbrs)] = nbrs

    positional_parameters = {"data": 0}
    named_parameters = {"context": context} if context is not None else {}

    result_vars = {}
    for var_name in data.data_vars:
        src = data[var_name]

        flat_indices = neighbor_matrix.ravel().copy()
        flat_indices[flat_indices < 0] = 0
        idx_da = xr.DataArray(flat_indices, dims="flat")
        flat_gathered = src.isel(**{dggs_dim: idx_da})

        non_dggs = tuple(d for d in flat_gathered.dims if d != "flat")
        non_dggs_sizes = tuple(flat_gathered.sizes[d] for d in non_dggs)
        gathered_np = flat_gathered.compute().values.reshape(
            *non_dggs_sizes, n_pix, max_nbrs
        )

        valid_mask = neighbor_matrix >= 0
        reduced_data = np.zeros(non_dggs_sizes + (n_pix,))
        for i in range(n_pix):
            vals = gathered_np[..., i, :][..., valid_mask[i]]
            val_flat = vals.ravel()
            result_val = process(
                val_flat,
                positional_parameters=positional_parameters,
                named_parameters=named_parameters,
            )
            reduced_data[..., i] = result_val

        dims = non_dggs + (dggs_dim,)
        coords = {}
        for d in non_dggs:
            if d in src.coords:
                coords[d] = src.coords[d]
        coords[dggs_dim] = src.coords[dggs_dim]

        reduced = xr.DataArray(reduced_data, dims=dims, coords=coords)
        result_vars[var_name] = reduced

    result = xr.Dataset(result_vars, coords=data.coords, attrs=data.attrs)
    return result


def mask_polygon_dggs(
    data: RasterCube,
    mask: dict,
    replacement: Optional[float] = None,
    inside: bool = False,
) -> RasterCube:
    dggs_dim = require_dggs_cube(data)

    if "lon" not in data.coords or "lat" not in data.coords:
        raise DimensionNotAvailable(
            "DGGS cube must have 'lat' and 'lon' coordinates for mask_polygon."
        )

    from openeo_processes_dask_slim.process_implementations.cubes._filter import (
        _load_geometries_dggs,
    )

    gdf = _load_geometries_dggs(mask)
    if gdf.crs is not None and not pyproj.crs.CRS(str(gdf.crs)).equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")

    union_geom = gdf.geometry.unary_union
    if union_geom is None or union_geom.is_empty:
        return data

    lon = np.asarray(data["lon"].data)
    lat = np.asarray(data["lat"].data)

    bounds = union_geom.bounds
    bbox_mask = (
        (lon >= bounds[0])
        & (lon <= bounds[2])
        & (lat >= bounds[1])
        & (lat <= bounds[3])
    )

    contained = np.array(
        [
            union_geom.contains(shapely.geometry.Point(lon[i], lat[i]))
            if bbox_mask[i]
            else False
            for i in range(len(lon))
        ]
    )

    if not inside:
        cell_mask = xr.DataArray(contained, dims=(dggs_dim,))
    else:
        cell_mask = xr.DataArray(~contained, dims=(dggs_dim,))

    replace_val = replacement if replacement is not None else np.nan
    result = data.where(cell_mask, replace_val)
    return result
