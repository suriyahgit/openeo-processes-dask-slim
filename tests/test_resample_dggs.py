import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.dggs import (
    dggs_to_raster,
    resample_dggs,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    OpenEOException,
)
from tests.mockdata import create_fake_healpix_cube


@pytest.fixture
def cube_n4():
    return create_fake_healpix_cube(nside=4, n_times=3, backend="numpy")


@pytest.fixture
def cube_n4_dask():
    return create_fake_healpix_cube(nside=4, n_times=3, backend="dask")


class TestResampleDggs:
    def test_downsample_n4_to_n2(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, order="nested", reducer="mean")
        assert isinstance(result, xr.Dataset)
        assert len(result.healpix_index) == 48
        assert result.attrs["healpix_nside"] == 2

    def test_downsample_n4_to_n1(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=1, reducer="mean")
        assert len(result.healpix_index) == 12

    def test_preserves_temporal(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="mean")
        assert result.dims["t"] == cube_n4.dims["t"]

    def test_preserves_bands(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="mean")
        for var in cube_n4.data_vars:
            assert var in result.data_vars

    def test_reducer_sum(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="sum")
        assert len(result.healpix_index) == 48

    def test_reducer_min(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="min")
        assert len(result.healpix_index) == 48

    def test_reducer_max(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="max")
        assert len(result.healpix_index) == 48

    def test_reducer_count(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="count")
        assert len(result.healpix_index) == 48

    def test_upsample_raises(self, cube_n4):
        with pytest.raises(NotImplementedError):
            resample_dggs(cube_n4, resolution=8, reducer="mean")

    def test_unsupported_reducer(self, cube_n4):
        with pytest.raises(ValueError):
            resample_dggs(cube_n4, resolution=2, reducer="invalid")

    def test_preserves_dask(self, cube_n4_dask):
        result = resample_dggs(cube_n4_dask, resolution=2, reducer="mean")
        for var_name in result.data_vars:
            assert isinstance(result[var_name].data, da.Array), (
                f"Variable {var_name} lost dask backing"
            )

    def test_updated_lat_lon(self, cube_n4):
        result = resample_dggs(cube_n4, resolution=2, reducer="mean")
        assert "lat" in result.coords
        assert "lon" in result.coords
        assert len(result.lat) == 48


class TestDggsToRaster:
    def test_converts_to_x_y(self, cube_n4):
        result = dggs_to_raster(cube_n4, resolution=10.0)
        assert isinstance(result, xr.Dataset)
        assert "x" in result.dims
        assert "y" in result.dims

    def test_preserves_temporal(self, cube_n4):
        result = dggs_to_raster(cube_n4, resolution=10.0)
        assert "t" in result.dims

    def test_preserves_bands(self, cube_n4):
        result = dggs_to_raster(cube_n4, resolution=10.0)
        for var in cube_n4.data_vars:
            assert var in result.data_vars

    def test_with_bbox(self, cube_n4):
        bbox = {"west": -45, "east": 45, "south": -45, "north": 45}
        result = dggs_to_raster(cube_n4, resolution=10.0, bbox=bbox)
        assert result.dims["x"] > 0
        assert result.dims["y"] > 0

    def test_preserves_dask(self, cube_n4_dask):
        result = dggs_to_raster(cube_n4_dask, resolution=10.0)
        for var_name in result.data_vars:
            assert isinstance(result[var_name].data, da.Array), (
                f"Variable {var_name} lost dask backing"
            )

    def test_unsupported_method(self, cube_n4):
        with pytest.raises(ValueError):
            dggs_to_raster(cube_n4, method="bilinear")
