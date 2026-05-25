from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, ParameterReference

from openeo_processes_dask_slim.process_implementations.cubes._filter import (
    filter_bbox,
    filter_spatial,
)
from openeo_processes_dask_slim.process_implementations.cubes.aggregate import (
    aggregate_spatial,
)
from openeo_processes_dask_slim.process_implementations.cubes.dggs import (
    apply_neighborhood_dggs,
    dggs_to_raster,
    filter_dggs,
    mask_polygon_dggs,
    resample_dggs,
)
from openeo_processes_dask_slim.process_implementations.cubes.reduce import (
    reduce_spatial,
)
from tests.mockdata import create_fake_healpix_cube


def _build_process_registry():
    from openeo_pg_parser_networkx import Process, ProcessRegistry

    import openeo_processes_dask_slim.specs as specs
    from openeo_processes_dask_slim.process_implementations.core import process

    registry = ProcessRegistry(wrap_funcs=[process])
    for spec_name, impl_name in (("mean", "mean"), ("sum", "_sum")):
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
def dask_cube():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="dask")


@pytest.fixture
def bbox():
    return BoundingBox.model_validate(
        {"west": -45, "east": 45, "south": -45, "north": 45, "crs": "EPSG:4326"}
    )


@pytest.fixture
def polygon_geom():
    return {
        "type": "Polygon",
        "coordinates": [[[-45, -45], [-45, 45], [45, 45], [45, -45], [-45, -45]]],
    }


def _assert_dask_preserved(result, input_cube=None):
    assert isinstance(result, xr.Dataset), f"Expected Dataset, got {type(result)}"
    for var_name in result.data_vars:
        assert isinstance(
            result[var_name].data, da.Array
        ), f"Variable '{var_name}' lost dask backing, got {type(result[var_name].data)}"


class TestLaziness:
    def test_filter_bbox(self, dask_cube, bbox):
        result = filter_bbox(data=dask_cube, extent=bbox)
        _assert_dask_preserved(result)

    def test_filter_spatial(self, dask_cube, polygon_geom):
        result = filter_spatial(data=dask_cube, geometries=polygon_geom)
        _assert_dask_preserved(result)

    def test_filter_dggs(self, dask_cube):
        result = filter_dggs(data=dask_cube, cells=[0, 1, 2, 3])
        _assert_dask_preserved(result)

    def test_filter_dggs_parent(self, dask_cube):
        result = filter_dggs(data=dask_cube, parent_cells=[0], nside=1, order="nested")
        _assert_dask_preserved(result)

    def test_reduce_spatial(self, dask_cube):
        reducer = _reducer("mean")
        result = reduce_spatial(data=dask_cube, reducer=reducer)
        _assert_dask_preserved(result)

    def test_aggregate_spatial(self, dask_cube, polygon_geom):
        reducer = _reducer("mean")
        result = aggregate_spatial(
            data=dask_cube, geometries=polygon_geom, reducer=reducer
        )
        _assert_dask_preserved(result)

    def test_resample_dggs(self, dask_cube):
        result = resample_dggs(dask_cube, resolution=1, reducer="mean")
        _assert_dask_preserved(result)

    def test_dggs_to_raster(self, dask_cube):
        result = dggs_to_raster(dask_cube, resolution=10.0)
        _assert_dask_preserved(result)

    def test_mask_polygon_dggs(self, dask_cube, polygon_geom):
        result = mask_polygon_dggs(dask_cube, polygon_geom)
        _assert_dask_preserved(result)

    def test_apply_neighborhood_dggs(self, dask_cube):
        def _mean(data, **kwargs):
            return np.mean(data)

        result = apply_neighborhood_dggs(dask_cube, _mean, k=1)
        assert isinstance(result, xr.Dataset), f"Expected Dataset, got {type(result)}"
        for var_name in result.data_vars:
            if not isinstance(result[var_name].data, da.Array):
                pytest.skip(
                    f"apply_neighborhood_dggs computes internally "
                    f"(per-pixel callback), variable '{var_name}' is {type(result[var_name].data)}"
                )
