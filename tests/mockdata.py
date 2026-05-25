import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

try:
    import astropy_healpix as ah

    HAS_ASTROPY_HEALPIX = True
except ImportError:
    HAS_ASTROPY_HEALPIX = False

logger = logging.getLogger(__name__)


def create_fake_rastercube(
    data,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    backend="numpy",
    chunks=("auto", "auto", "auto", -1),
    as_dataset=True,
):
    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    x_coords = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        step=len_x / data.shape[0],
    )
    y_coords = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        step=len_y / data.shape[1],
    )

    # This line raises a deprecation warning, which according to this thread
    # will never actually be deprecated:
    # https://github.com/numpy/numpy/issues/23904
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t_coords = pd.date_range(
            start=np.datetime64(temporal_extent.root[0].root),
            end=np.datetime64(temporal_extent.root[1].root),
            periods=data.shape[2],
        ).values

    coords = {"x": x_coords, "y": y_coords, "t": t_coords, "bands": bands}

    raster_cube = xr.DataArray(
        data=data,
        coords=coords,
        attrs={"crs": spatial_extent.crs},
    )
    import odc.geo.xr

    raster_cube = odc.geo.xr.assign_crs(raster_cube, crs=spatial_extent.crs)

    if "dask" in backend:
        import dask.array as da

        raster_cube.data = da.from_array(raster_cube.data, chunks=chunks)

    if as_dataset:
        raster_cube = raster_cube.to_dataset(dim="bands")

    return raster_cube


def create_fake_healpix_cube(
    nside=2,
    n_times=3,
    n_bands=1,
    band_names=None,
    seed=42,
    backend="numpy",
    chunks=("auto", -1),
):
    if not HAS_ASTROPY_HEALPIX:
        raise ImportError(
            "astropy-healpix is required to create synthetic HEALPix cubes. "
            "Install it with: pip install astropy-healpix"
        )

    rng = np.random.default_rng(seed)
    npix = ah.nside_to_npix(nside)
    healpix_indices = np.arange(npix)

    lon_rad, lat_rad = ah.healpix_to_lonlat(
        healpix_indices, nside, order="ring"
    )
    lat_vals = np.rad2deg(np.asarray(lat_rad))
    lon_vals = np.rad2deg(np.asarray(lon_rad))

    t_coords = pd.date_range(start="2020-01-01", periods=n_times, freq="D").values

    if band_names is None:
        band_names = [f"band_{i}" for i in range(n_bands)]

    data_vars = {}
    for band_name in band_names:
        shape = (n_times, npix)
        band_data = rng.random(shape)
        if "dask" in backend:
            import dask.array as da

            band_data = da.from_array(band_data, chunks=chunks)
        data_vars[band_name] = xr.Variable(
            ("t", "healpix_index"),
            band_data,
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "t": t_coords,
            "healpix_index": healpix_indices,
            "lat": xr.Variable("healpix_index", lat_vals),
            "lon": xr.Variable("healpix_index", lon_vals),
        },
        attrs={
            "crs": f"healpix:{nside}",
            "healpix_nside": nside,
            "healpix_order": "ring",
        },
    )

    return ds
