from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask_slim.process_implementations.cubes.aggregate import (
    aggregate_spatial,
)
from openeo_processes_dask_slim.process_implementations.cubes.reduce import (
    reduce_spatial,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    NoDataAvailable,
)
from tests.mockdata import create_fake_healpix_cube


def _build_process_registry():
    from openeo_pg_parser_networkx import Process, ProcessRegistry

    import openeo_processes_dask_slim.specs as specs
    from openeo_processes_dask_slim.process_implementations.core import process

    registry = ProcessRegistry(wrap_funcs=[process])
    for spec_name, impl_name in (
        ("mean", "mean"),
        ("_sum", "_sum"),
        ("_min", "_min"),
        ("_max", "_max"),
    ):
        spec = getattr(specs, spec_name, None)
        impl = getattr(
            __import__(
                "openeo_processes_dask_slim.process_implementations.math",
                fromlist=[impl_name],
            ),
            impl_name,
            None,
        )
        if impl:
            registry[spec_name.removeprefix("_")] = Process(
                spec=spec, implementation=impl
            )
    return registry


registry = _build_process_registry()


def _reducer(name, **extra):
    return partial(
        registry[name].implementation,
        data=ParameterReference(from_parameter="data"),
        **extra,
    )


@pytest.fixture
def cube_numpy():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="numpy")


@pytest.fixture
def cube_dask():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="dask")


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


class TestReduceSpatialDggs:
    def test_reduces_over_healpix_dim(self, cube_numpy):
        reducer = _reducer("mean")
        result = reduce_spatial(data=cube_numpy, reducer=reducer)
        assert "healpix_index" not in result.dims
        assert "t" in result.dims

    def test_preserves_bands(self, cube_numpy):
        reducer = _reducer("mean")
        result = reduce_spatial(data=cube_numpy, reducer=reducer)
        for var in cube_numpy.data_vars:
            assert var in result.data_vars

    def test_reduce_sum(self, cube_numpy):
        reducer = _reducer("sum")
        result = reduce_spatial(data=cube_numpy, reducer=reducer)
        assert "t" in result.dims

    def test_preserves_dask(self, cube_dask):
        reducer = _reducer("mean")
        result = reduce_spatial(data=cube_dask, reducer=reducer)
        for var_name in result.data_vars:
            assert isinstance(
                result[var_name].data, da.Array
            ), f"Variable {var_name} lost dask backing"


class TestAggregateSpatialDggs:
    def test_single_polygon(self, cube_numpy, polygon_geometry):
        reducer = _reducer("mean")
        result = aggregate_spatial(
            data=cube_numpy, geometries=polygon_geometry, reducer=reducer
        )
        assert isinstance(result, xr.Dataset)
        assert "geometry" in result.dims

    def test_preserves_bands(self, cube_numpy, polygon_geometry):
        reducer = _reducer("mean")
        result = aggregate_spatial(
            data=cube_numpy, geometries=polygon_geometry, reducer=reducer
        )
        for var in cube_numpy.data_vars:
            assert var in result.data_vars

    def test_reducer_sum(self, cube_numpy, polygon_geometry):
        reducer = _reducer("sum")
        result = aggregate_spatial(
            data=cube_numpy, geometries=polygon_geometry, reducer=reducer
        )
        assert "geometry" in result.dims

    def test_reducer_min(self, cube_numpy, polygon_geometry):
        reducer = _reducer("min")
        result = aggregate_spatial(
            data=cube_numpy, geometries=polygon_geometry, reducer=reducer
        )
        assert "geometry" in result.dims

    def test_reducer_max(self, cube_numpy, polygon_geometry):
        reducer = _reducer("max")
        result = aggregate_spatial(
            data=cube_numpy, geometries=polygon_geometry, reducer=reducer
        )
        assert "geometry" in result.dims

    def test_single_polygon_dask(self, cube_dask, polygon_geometry):
        reducer = _reducer("mean")
        result = aggregate_spatial(
            data=cube_dask, geometries=polygon_geometry, reducer=reducer
        )
        for var_name in result.data_vars:
            assert isinstance(
                result[var_name].data, da.Array
            ), f"Variable {var_name} lost dask backing"

    def test_empty_geometry_raises(self, cube_numpy):
        reducer = _reducer("mean")
        far_geom = {
            "type": "Polygon",
            "coordinates": [
                [[200, -80], [200, -70], [210, -70], [210, -80], [200, -80]]
            ],
        }
        with pytest.raises(NoDataAvailable):
            aggregate_spatial(data=cube_numpy, geometries=far_geom, reducer=reducer)

    def test_single_polygon_geom(self, cube_numpy):
        reducer = _reducer("mean")
        geom = {
            "type": "Polygon",
            "coordinates": [[[-45, -45], [-45, 45], [45, 45], [45, -45], [-45, -45]]],
        }
        result = aggregate_spatial(data=cube_numpy, geometries=geom, reducer=reducer)
        assert "geometry" in result.dims
        assert len(result.geometry) == 1

    def test_multi_feature_feature_collection(self, cube_numpy):
        reducer = _reducer("mean")
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-1, 14], [-1, 25], [6, 25], [6, 14], [-1, 14]]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[19, 34], [19, 46], [26, 46], [26, 34], [19, 34]]
                        ],
                    },
                },
            ],
        }
        result = aggregate_spatial(data=cube_numpy, geometries=fc, reducer=reducer)
        assert len(result.geometry) == 2

    def test_multi_feature_preserves_order(self, cube_numpy):
        reducer = _reducer("mean")
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "a"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[19, 34], [19, 46], [26, 46], [26, 34], [19, 34]]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"name": "b"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-1, 14], [-1, 25], [6, 25], [6, 14], [-1, 14]]
                        ],
                    },
                },
            ],
        }
        result = aggregate_spatial(data=cube_numpy, geometries=fc, reducer=reducer)
        assert len(result.geometry) == 2
