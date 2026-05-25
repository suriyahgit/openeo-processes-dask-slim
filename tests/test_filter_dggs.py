import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes._filter import (
    filter_bbox,
    filter_spatial,
)
from openeo_processes_dask_slim.process_implementations.cubes.dggs import filter_dggs
from openeo_processes_dask_slim.process_implementations.exceptions import (
    NoDataAvailable,
)
from tests.mockdata import create_fake_healpix_cube


@pytest.fixture
def cube_numpy():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="numpy")


@pytest.fixture
def cube_dask():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="dask")


@pytest.fixture
def bbox_wgs84():
    from openeo_pg_parser_networkx.pg_schema import BoundingBox

    return BoundingBox.model_validate(
        {"west": -45, "east": 45, "south": -45, "north": 45, "crs": "EPSG:4326"}
    )


@pytest.fixture
def polygon_geometry():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-45, -45], [-45, 45], [45, 45], [45, -45], [-45, -45]]
                    ],
                },
            }
        ],
    }


class TestFilterBboxDggs:
    def test_filters_dggs_cube(self, cube_numpy, bbox_wgs84):
        result = filter_bbox(data=cube_numpy, extent=bbox_wgs84)
        assert isinstance(result, xr.Dataset)
        assert "healpix_index" in result.dims
        assert len(result.healpix_index) < len(cube_numpy.healpix_index)

    def test_preserves_temporal_dim(self, cube_numpy, bbox_wgs84):
        result = filter_bbox(data=cube_numpy, extent=bbox_wgs84)
        assert "t" in result.dims
        assert len(result.t) == len(cube_numpy.t)

    def test_preserves_attrs(self, cube_numpy, bbox_wgs84):
        result = filter_bbox(data=cube_numpy, extent=bbox_wgs84)
        assert result.attrs["crs"] == cube_numpy.attrs["crs"]
        assert result.attrs["healpix_nside"] == cube_numpy.attrs["healpix_nside"]

    def test_preserves_dask_laziness(self, cube_dask, bbox_wgs84):
        result = filter_bbox(data=cube_dask, extent=bbox_wgs84)
        for var_name in result.data_vars:
            assert isinstance(
                result[var_name].data, da.Array
            ), f"Variable {var_name} lost dask backing"

    def test_raises_on_no_cells(self, cube_numpy):
        from openeo_pg_parser_networkx.pg_schema import BoundingBox

        far_bbox = BoundingBox.model_validate(
            {"west": 200, "east": 210, "south": -80, "north": -70, "crs": "EPSG:4326"}
        )
        with pytest.raises(NoDataAvailable):
            filter_bbox(data=cube_numpy, extent=far_bbox)

    def test_planar_cube_unaffected(self, bounding_box, temporal_interval):
        from tests.mockdata import create_fake_rastercube

        data = (
            np.random.default_rng(42)
            .integers(-100, 100, size=(30, 30, 30, 1))
            .astype(np.uint8)
        )
        cube = create_fake_rastercube(
            data=data,
            spatial_extent=bounding_box,
            temporal_extent=temporal_interval,
            bands=["B02"],
            backend="numpy",
        )
        result = filter_bbox(data=cube, extent=bounding_box)
        assert "x" in result.dims
        assert "y" in result.dims


class TestFilterSpatialDggs:
    def test_filters_by_polygon(self, cube_numpy, polygon_geometry):
        result = filter_spatial(data=cube_numpy, geometries=polygon_geometry)
        assert isinstance(result, xr.Dataset)
        assert "healpix_index" in result.dims
        assert len(result.healpix_index) < len(cube_numpy.healpix_index)

    def test_preserves_temporal(self, cube_numpy, polygon_geometry):
        result = filter_spatial(data=cube_numpy, geometries=polygon_geometry)
        assert len(result.t) == len(cube_numpy.t)

    def test_preserves_dask(self, cube_dask, polygon_geometry):
        result = filter_spatial(data=cube_dask, geometries=polygon_geometry)
        for var_name in result.data_vars:
            assert isinstance(result[var_name].data, da.Array)

    def test_single_polygon(self, cube_numpy):
        geom = {
            "type": "Polygon",
            "coordinates": [[[-45, -45], [-45, 45], [45, 45], [45, -45], [-45, -45]]],
        }
        result = filter_spatial(data=cube_numpy, geometries=geom)
        assert len(result.healpix_index) > 0

    def test_empty_result_raises(self, cube_numpy):
        far_geom = {
            "type": "Polygon",
            "coordinates": [
                [[200, -80], [200, -70], [210, -70], [210, -80], [200, -80]]
            ],
        }
        with pytest.raises(NoDataAvailable):
            filter_spatial(data=cube_numpy, geometries=far_geom)


class TestFilterDggs:
    def test_selects_by_cells(self, cube_numpy):
        cell_indices = [0, 1, 2, 3]
        result = filter_dggs(data=cube_numpy, cells=cell_indices)
        assert len(result.healpix_index) == 4
        assert list(result.healpix_index.values) == cell_indices

    def test_selects_by_parent_cells(self, cube_numpy):
        result = filter_dggs(data=cube_numpy, parent_cells=[0], nside=1, order="nested")
        assert len(result.healpix_index) == 4
        assert set(result.healpix_index.values) == {0, 1, 2, 3}

    def test_requires_at_least_one_param(self, cube_numpy):
        with pytest.raises(ValueError):
            filter_dggs(data=cube_numpy)

    def test_preserves_dask_laziness(self, cube_dask):
        result = filter_dggs(data=cube_dask, cells=[0, 1, 2, 3])
        for var_name in result.data_vars:
            assert isinstance(result[var_name].data, da.Array)

    def test_preserves_attrs(self, cube_numpy):
        result = filter_dggs(data=cube_numpy, cells=[0, 5, 10])
        assert result.attrs["healpix_nside"] == 2

    def test_cells_and_parents_union(self, cube_numpy):
        result = filter_dggs(
            data=cube_numpy, cells=[10, 20], parent_cells=[0], nside=1, order="nested"
        )
        assert len(result.healpix_index) == 6
