import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.dggs import (
    get_dggs_crs,
    get_dggs_dim,
    get_dggs_grid_system,
    get_dggs_order,
    get_dggs_resolution,
    is_dggs_cube,
    is_healpix_cube,
    require_dggs_cube,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
)
from tests.mockdata import create_fake_healpix_cube


@pytest.fixture
def dggs_cube_numpy():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="numpy")


@pytest.fixture
def dggs_cube_dask():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="dask")


class TestDggsFixtures:
    def test_numpy_cube_structure(self, dggs_cube_numpy):
        ds = dggs_cube_numpy
        assert isinstance(ds, xr.Dataset)
        assert "t" in ds.dims
        assert "healpix_index" in ds.dims
        assert len(ds.healpix_index) == 48
        assert not isinstance(ds["band_0"].data, da.Array)

    def test_dask_cube_dask_backed(self, dggs_cube_dask):
        ds = dggs_cube_dask
        for var_name in ds.data_vars:
            assert isinstance(ds[var_name].data, da.Array), (
                f"Variable {var_name} is not dask-backed"
            )

    def test_cell_center_coords(self, dggs_cube_numpy):
        ds = dggs_cube_numpy
        assert "lat" in ds.coords
        assert "lon" in ds.coords
        assert ds.coords["lat"].dims == ("healpix_index",)
        assert ds.coords["lon"].dims == ("healpix_index",)
        assert len(ds.coords["lat"]) == 48

    def test_legacy_healpix_attrs(self, dggs_cube_numpy):
        ds = dggs_cube_numpy
        assert ds.attrs["crs"] == "healpix:2"
        assert ds.attrs["healpix_nside"] == 2
        assert ds.attrs["healpix_order"] == "ring"


class TestIsDggsCube:
    def test_detects_healpix_cube_as_dggs(self, dggs_cube_numpy):
        assert is_dggs_cube(dggs_cube_numpy) is True

    def test_detects_healpix_cube(self, dggs_cube_numpy):
        assert is_healpix_cube(dggs_cube_numpy) is True

    def test_rejects_planar_cube(self, bounding_box, temporal_interval):
        from tests.mockdata import create_fake_rastercube

        data = np.random.default_rng(42).integers(-100, 100, size=(30, 30, 30, 1)).astype(np.uint8)
        cube = create_fake_rastercube(
            data=data,
            spatial_extent=bounding_box,
            temporal_extent=temporal_interval,
            bands=["B02"],
            backend="numpy",
        )
        assert is_dggs_cube(cube) is False
        assert is_healpix_cube(cube) is False

    def test_rejects_empty_dataset(self):
        ds = xr.Dataset({"a": xr.Variable("x", [1, 2, 3])})
        assert is_dggs_cube(ds) is False

    def test_detects_dggs_via_attrs(self):
        ds = xr.Dataset(
            {"a": xr.Variable("cell_id", [1, 2, 3])},
            attrs={
                "dggs_grid_system": "healpix",
                "dggs_cell_id_dim": "cell_id",
                "dggs_resolution": 2,
            },
        )
        assert is_dggs_cube(ds) is True


class TestDggsHelpers:
    def test_get_dggs_dim(self, dggs_cube_numpy):
        assert get_dggs_dim(dggs_cube_numpy) == "healpix_index"

    def test_get_dggs_dim_none(self):
        ds = xr.Dataset({"a": xr.Variable("x", [1, 2, 3])})
        assert get_dggs_dim(ds) is None

    def test_get_dggs_grid_system_legacy(self, dggs_cube_numpy):
        assert get_dggs_grid_system(dggs_cube_numpy) is None

    def test_get_dggs_grid_system_from_attrs(self):
        ds = xr.Dataset(
            {"a": xr.Variable("cell_id", [1, 2, 3])},
            attrs={"dggs_grid_system": "healpix", "dggs_cell_id_dim": "cell_id"},
        )
        assert get_dggs_grid_system(ds) == "healpix"

    def test_get_dggs_resolution_legacy(self, dggs_cube_numpy):
        assert get_dggs_resolution(dggs_cube_numpy) == 2

    def test_get_dggs_order_legacy(self, dggs_cube_numpy):
        assert get_dggs_order(dggs_cube_numpy) == "ring"

    def test_get_dggs_crs(self, dggs_cube_numpy):
        assert get_dggs_crs(dggs_cube_numpy) == "healpix:2"

    def test_require_dggs_cube_valid(self, dggs_cube_numpy):
        dim = require_dggs_cube(dggs_cube_numpy)
        assert dim == "healpix_index"

    def test_require_dggs_cube_invalid(self):
        ds = xr.Dataset({"a": xr.Variable("x", [1, 2, 3])})
        with pytest.raises(DimensionNotAvailable):
            require_dggs_cube(ds)


class TestDggsAccessor:
    def test_spatial_dims(self, dggs_cube_numpy):
        assert dggs_cube_numpy.openeo.spatial_dims == ("healpix_index",)

    def test_x_dim_none(self, dggs_cube_numpy):
        assert dggs_cube_numpy.openeo.x_dim is None

    def test_y_dim_none(self, dggs_cube_numpy):
        assert dggs_cube_numpy.openeo.y_dim is None

    def test_temporal_dims(self, dggs_cube_numpy):
        assert "t" in dggs_cube_numpy.openeo.temporal_dims

    def test_dggs_attrs_detection(self):
        ds = xr.Dataset(
            {"a": xr.Variable("cell_id", [1, 2, 3])},
            coords={"cell_id": [10, 20, 30]},
            attrs={
                "dggs_grid_system": "healpix",
                "dggs_cell_id_dim": "cell_id",
                "dggs_resolution": 2,
                "dggs_order": "ring",
                "crs": "healpix:2",
            },
        )
        assert ds.openeo.spatial_dims == ("cell_id",)
        assert ds.openeo.x_dim is None
        assert ds.openeo.y_dim is None

    def test_planar_cube_unaffected(self, bounding_box, temporal_interval):
        from tests.mockdata import create_fake_rastercube

        data = np.random.default_rng(42).integers(-100, 100, size=(30, 30, 30, 1)).astype(np.uint8)
        cube = create_fake_rastercube(
            data=data,
            spatial_extent=bounding_box,
            temporal_extent=temporal_interval,
            bands=["B02"],
            backend="numpy",
        )
        assert "x" in cube.openeo.spatial_dims
        assert "y" in cube.openeo.spatial_dims
        assert cube.openeo.x_dim is not None
        assert cube.openeo.y_dim is not None
