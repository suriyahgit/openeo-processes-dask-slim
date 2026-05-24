from functools import partial

import dask
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask_slim.process_implementations import merge_cubes
from openeo_processes_dask_slim.process_implementations.cubes.merge import (
    NEW_DIM_COORDS,
    NEW_DIM_NAME,
)
from openeo_processes_dask_slim.process_implementations.exceptions import (
    OverlapResolverMissing,
)
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_1(temporal_interval, bounding_box, random_raster_data):
    """See Example 1 from https://processes.openeo.org/#merge_cubes."""
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "--324"],
        backend="dask",
    )

    cube_1 = origin_cube.drop_vars(["B04", "--324"])
    cube_2 = origin_cube.drop_vars(["B02", "B03"])

    merged_cube = merge_cubes(cube_1, cube_2)
    for var in merged_cube.data_vars.values():
        assert isinstance(var.data, dask.array.Array)

    xr.testing.assert_equal(merged_cube, origin_cube)


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_2(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
        backend="dask",
    )

    cube_1 = origin_cube.drop_vars("B03")
    cube_2 = origin_cube.drop_vars("B01")

    with pytest.raises(OverlapResolverMissing):
        merge_cubes(cube_1, cube_2)

    overlap_resolver = partial(
        process_registry["add"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=ParameterReference(from_parameter="y"),
    )
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    for var in merged_cube.data_vars.values():
        assert isinstance(var.data, dask.array.Array)

    xr.testing.assert_equal(merged_cube["B02"] / 2, origin_cube["B02"])


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_3(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    # This is basically broadcasting the smaller datacube and then applying the overlap resolver.
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
        backend="dask",
    )

    cube_1 = origin_cube
    cube_2 = origin_cube

    # If no overlap reducer is provided, then simply concatenate along a new dimension
    merged_cube = merge_cubes(cube_1, cube_2)
    expected_result = xr.concat([cube_1, cube_2], dim=NEW_DIM_NAME).reindex(
        {NEW_DIM_NAME: NEW_DIM_COORDS}
    )
    xr.testing.assert_equal(merged_cube, expected_result)

    # If an overlap reducer is provided, then reduce per pixel
    merged_cube = merge_cubes(
        cube_1,
        cube_2,
        partial(
            process_registry["add"].implementation,
            x=ParameterReference(from_parameter="x"),
            y=ParameterReference(from_parameter="y"),
        ),
    )
    for var in merged_cube.data_vars.values():
        assert isinstance(var.data, dask.array.Array)

    xr.testing.assert_equal(merged_cube, cube_1 * 2)


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_type_4(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    # This is basically broadcasting the smaller datacube and then applying the overlap resolver.
    cube_1 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02", "B03"],
        backend="dask",
    )

    cube_2_vars = {}
    for var in cube_1.data_vars:
        cube_2_vars[var] = xr.DataArray(
            np.ones((len(cube_1["x"]), len(cube_1["y"]))),
            dims=["x", "y"],
            coords={"x": cube_1.coords["x"], "y": cube_1.coords["y"]},
        )
    cube_2 = xr.Dataset(cube_2_vars)

    with pytest.raises(OverlapResolverMissing):
        merge_cubes(cube_1, cube_2)

    overlap_resolver = partial(
        process_registry["add"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=ParameterReference(from_parameter="y"),
    )
    merged_cube_1 = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    merged_cube_2 = merge_cubes(cube_2, cube_1, overlap_resolver=overlap_resolver)

    for var in merged_cube_1.data_vars.values():
        assert isinstance(var.data, dask.array.Array)
    xr.testing.assert_equal(merged_cube_1, cube_1 + 1)

    for var in merged_cube_2.data_vars.values():
        assert isinstance(var.data, dask.array.Array)
    xr.testing.assert_equal(merged_cube_2, cube_1 + 1)


@pytest.mark.parametrize("size", [(6, 5, 4, 1)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_conflicting_coords(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    # See https://github.com/Open-EO/openeo-processes-dask/pull/148 for why is is necessary
    # This is basically broadcasting the smaller datacube and then applying the overlap resolver.
    cube_1 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01"],
        backend="dask",
    )
    cube_1 = cube_1.assign_coords({"s2:processing_baseline": "05.8"})
    cube_2 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )
    cube_2 = cube_2.assign_coords({"s2:processing_baseline": "05.9"})

    merged_cube_1 = merge_cubes(cube_1, cube_2)

    for var in merged_cube_1.data_vars.values():
        assert isinstance(var.data, dask.array.Array)


def test_merge_float_coord_alignment(bounding_box, temporal_interval):
    shape = (10, 10, 5, 1)
    data = np.random.rand(*shape).astype(np.float32)

    cube_a = create_fake_rastercube(
        data=data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B04"],
        backend="dask",
    )

    cube_b = cube_a.copy(deep=True)
    cube_b = cube_b.assign_coords(
        x=(cube_b["x"] + 1e-6),
        y=(cube_b["y"] - 1e-6),
    )

    merged = merge_cubes(cube_a, cube_b)
    assert isinstance(merged, xr.Dataset)


@pytest.mark.parametrize("size", [(6, 5, 4, 2)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_dataset(temporal_interval, bounding_box, random_raster_data):
    cube_a = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03"],
        backend="dask",
        as_dataset=True,
    )

    rng = np.random.default_rng(99)
    cube_b_data = rng.integers(-100, 100, size=(6, 5, 4, 2)).astype(np.float64)
    cube_b = create_fake_rastercube(
        data=cube_b_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    merged = merge_cubes(cube_a, cube_b)
    assert isinstance(merged, xr.Dataset)
    assert set(merged.data_vars) == {"B02", "B03", "B04", "B08"}


@pytest.mark.parametrize("size", [(6, 5, 4, 3)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_preserves_var_order(
    temporal_interval, bounding_box, random_raster_data
):
    cube_a = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B08", "B02", "B03"],
        backend="numpy",
        as_dataset=True,
    )
    cube_a = cube_a[["B08", "B02", "B03"]]

    cube_b = create_fake_rastercube(
        data=np.random.default_rng(42)
        .integers(-100, 100, size=(6, 5, 4, 1))
        .astype(np.float64),
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B04"],
        backend="numpy",
        as_dataset=True,
    )

    merged = merge_cubes(cube_a, cube_b)
    assert list(merged.data_vars) == ["B08", "B02", "B03", "B04"]


@pytest.mark.parametrize("size", [(6, 5, 4, 2)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_preserves_var_attrs(
    temporal_interval, bounding_box, random_raster_data
):
    cube1 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03"],
        backend="numpy",
        as_dataset=True,
    )
    cube2_data = (
        np.random.default_rng(77)
        .integers(-100, 100, size=(6, 5, 4, 2))
        .astype(np.float64)
    )
    cube2 = create_fake_rastercube(
        data=cube2_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B04", "B08"],
        backend="numpy",
        as_dataset=True,
    )

    cube1["B02"].attrs["description"] = "Band 2"
    cube1["B03"].attrs["description"] = "Band 3"
    cube2["B04"].attrs["description"] = "Band 4"
    cube2["B08"].attrs["description"] = "Band 8"

    merged = merge_cubes(cube1, cube2)
    assert merged["B02"].attrs.get("description") == "Band 2"
    assert merged["B03"].attrs.get("description") == "Band 3"
    assert merged["B04"].attrs.get("description") == "Band 4"
    assert merged["B08"].attrs.get("description") == "Band 8"


def test_merge_cubes_disjoint_coords():
    ds1 = xr.Dataset(
        {"B02": xr.DataArray(np.ones(2), dims=["t"], coords={"t": [0, 1]})}
    )
    ds2 = xr.Dataset(
        {"B03": xr.DataArray(np.ones(2) * 2, dims=["t"], coords={"t": [2, 3]})}
    )
    merged = merge_cubes(ds1, ds2)
    assert list(merged.t.values) == [0, 1, 2, 3]
    assert merged["B03"].sel(t=[2, 3]).notnull().all()
    assert merged["B03"].sel(t=[0, 1]).isnull().all()
    assert merged["B02"].sel(t=[0, 1]).notnull().all()
    assert merged["B02"].sel(t=[2, 3]).isnull().all()


@pytest.mark.parametrize("size", [(6, 5, 4, 2)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes_preserves_dask(
    temporal_interval, bounding_box, random_raster_data
):
    import dask

    cube1 = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03"],
        backend="dask",
        as_dataset=True,
    )
    cube2_data = (
        np.random.default_rng(55)
        .integers(-100, 100, size=(6, 5, 4, 2))
        .astype(np.float64)
    )
    cube2 = create_fake_rastercube(
        data=cube2_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    merged = merge_cubes(cube1, cube2)
    for var in merged.data_vars.values():
        assert isinstance(var.data, dask.array.Array)


def test_merge_cubes_disjoint_coords_alignment():
    """P0.3: merge_cubes aligns float coordinates for disjoint variables."""
    ds1 = xr.Dataset(
        {"B02": xr.DataArray(np.ones((2,)), dims=["x"], coords={"x": [0.0, 1.0]})}
    )
    ds2 = xr.Dataset(
        {
            "B03": xr.DataArray(
                np.ones((2,)) * 2, dims=["x"], coords={"x": [1e-7, 1.0000001]}
            )
        }
    )
    result = merge_cubes(ds1, ds2)
    assert list(result.data_vars) == ["B02", "B03"]
    assert len(result["x"]) == 2, f"Expected 2 coords, got {len(result['x'])}"
    np.testing.assert_allclose(result["x"].values, [0.0, 1.0], atol=1e-6)
    assert not result["B03"].isnull().any(), "B03 should not have NaN after alignment"
