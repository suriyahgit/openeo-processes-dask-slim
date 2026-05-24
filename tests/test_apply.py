from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask_slim.process_implementations.cubes.apply import (
    apply,
    apply_dimension,
    apply_kernel,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process = partial(
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="x"),
    )

    output_cube = apply(data=input_cube, process=_process)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube + 1),
    )

    xr.testing.assert_equal(output_cube, input_cube + 1)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_add(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process = partial(
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="data"),
    )

    # Target dimension is null and therefore defaults to the source dimension
    output_cube_same_pixels = apply_dimension(
        data=input_cube, process=_process, dimension="x"
    )

    for var_name in input_cube.data_vars:
        np.testing.assert_allclose(
            output_cube_same_pixels[var_name].data,
            (input_cube[var_name] + 1).data,
        )


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_ordering_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process_order = partial(
        process_registry["order"].implementation,
        data=ParameterReference(from_parameter="data"),
        nodata=True,
    )

    output_cube_order = apply_dimension(
        data=input_cube,
        process=_process_order,
        dimension="x",
        target_dimension="target",
    )

    for var_name in input_cube.data_vars:
        var_data = input_cube[var_name].data.compute()
        expected_order = np.argsort(var_data, kind="mergesort", axis=0)
        assert isinstance(output_cube_order[var_name].data, da.Array)
        np.testing.assert_array_equal(
            output_cube_order[var_name].data.compute(), expected_order
        )

    _process_rearrange = partial(
        process_registry["rearrange"].implementation,
        data=ParameterReference(from_parameter="data"),
        order=da.from_array(np.array([0, 1, 2, 3])),
    )

    output_cube_rearrange = apply_dimension(
        data=input_cube, process=_process_rearrange, dimension="x", target_dimension="x"
    )

    assert list(output_cube_rearrange.dims) == list(input_cube.dims)
    for var in output_cube_rearrange.data_vars.values():
        assert isinstance(var.data, da.Array)

    _process_sort = partial(
        process_registry["sort"].implementation,
        data=ParameterReference(from_parameter="data"),
        nodata=True,
    )

    output_cube_sort = apply_dimension(
        data=input_cube, process=_process_sort, dimension="x", target_dimension="target"
    )

    for var_name in input_cube.data_vars:
        var_data = input_cube[var_name].data.compute()
        expected_sort = np.sort(var_data, axis=0)
        assert isinstance(output_cube_sort[var_name].data, da.Array)
        np.testing.assert_array_equal(
            output_cube_sort[var_name].data.compute(), expected_sort
        )

        expected_order = np.argsort(var_data, kind="mergesort", axis=0)
        rearrange_by_expected = np.take_along_axis(
            var_data, indices=expected_order, axis=0
        )
        np.testing.assert_array_equal(
            output_cube_sort[var_name].data.compute(), rearrange_by_expected
        )


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_quantile_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )
    probability = 4

    _process_quantile = partial(
        process_registry["quantiles"].implementation,
        data=ParameterReference(from_parameter="data"),
        probabilities=probability,
    )

    output_cube_quantile = apply_dimension(
        data=input_cube,
        process=_process_quantile,
        dimension="t",
    )
    for var in output_cube_quantile.data_vars.values():
        assert var.shape == (6, 5, probability - 1)


@pytest.mark.parametrize("size", [(6, 5, 10, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_interpolate_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    data_with_nan = np.array(random_raster_data, copy=True)
    data_with_nan[3, 2, 5, 0] = np.nan
    input_cube = create_fake_rastercube(
        data=data_with_nan,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="numpy",
        as_dataset=True,
    )
    band_names = list(input_cube.data_vars)

    _process_interpolate = partial(
        process_registry["array_interpolate_linear"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = apply_dimension(
        data=input_cube,
        process=_process_interpolate,
        dimension="t",
    )
    assert output_cube[band_names[0]].data.shape == input_cube[band_names[0]].data.shape


@pytest.mark.parametrize("size", [(6, 5, 10, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_modify_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process_modify = partial(
        process_registry["array_modify"].implementation,
        data=ParameterReference(from_parameter="data"),
        values=[2, 3],
        index=3,
    )

    output_cube = apply_dimension(
        data=input_cube,
        process=_process_modify,
        dimension="bands",
    )
    assert len(output_cube.data_vars) == 5


@pytest.mark.parametrize("size", [(6, 5, 10, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_filter_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _condition = partial(
        process_registry["gt"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=10,
    )

    _process_filter = partial(
        process_registry["array_filter"].implementation,
        data=ParameterReference(from_parameter="data"),
        condition=_condition,
    )

    output_cube = apply_dimension(
        data=input_cube,
        process=_process_filter,
        dimension="bands",
    )
    for var_name in output_cube.data_vars:
        assert output_cube[var_name].shape <= input_cube[var_name].shape


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_kernel(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    # Following kernel should leave cube unchanged
    kernel = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    output_cube = apply_kernel(data=input_cube, kernel=kernel)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=input_cube,
    )

    xr.testing.assert_equal(output_cube, input_cube)


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cumsum_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process_cumsum = partial(
        process_registry["cumsum"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cumsum = apply_dimension(
        data=input_cube,
        process=_process_cumsum,
        dimension="t",
    ).compute()

    original_abs_sum = sum(
        np.sum(np.abs(input_cube[var].data)) for var in input_cube.data_vars
    )
    cumsum_total = sum(
        np.sum(np.abs(output_cube_cumsum[var].data))
        for var in output_cube_cumsum.data_vars
    )

    assert cumsum_total >= original_abs_sum

    band_names = list(input_cube.data_vars)
    for name in band_names:
        data = input_cube[name].values
        data[:, :, 15] = np.nan
        input_cube[name] = (input_cube[name].dims, data)

    _process_cumsum_with_nan = partial(
        process_registry["cumsum"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cumsum_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cumsum_with_nan,
        dimension="t",
    ).compute()

    first_var = list(output_cube_cumsum_with_nan.data_vars.values())[0]
    assert np.isnan(first_var.data[0, 0, 20])


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cumproduct_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process_cumsum = partial(
        process_registry["cumproduct"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cumprod = apply_dimension(
        data=input_cube,
        process=_process_cumsum,
        dimension="t",
    ).compute()

    original_abs_prod = sum(
        np.sum(np.abs(np.nan_to_num(input_cube[var].data)))
        for var in input_cube.data_vars
    )
    cumprod_total = sum(
        np.sum(np.abs(np.nan_to_num(output_cube_cumprod[var].data)))
        for var in output_cube_cumprod.data_vars
    )

    assert cumprod_total >= original_abs_prod

    band_names = list(input_cube.data_vars)
    for name in band_names:
        data = input_cube[name].values
        data[:, :, 15] = np.nan
        input_cube[name] = (input_cube[name].dims, data)

    _process_cumprod_with_nan = partial(
        process_registry["cumproduct"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cumprod_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cumprod_with_nan,
        dimension="t",
    ).compute()

    first_var = list(output_cube_cumprod_with_nan.data_vars.values())[0]
    assert np.isnan(first_var.data[0, 0, 20])


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cummax_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process_cummax = partial(
        process_registry["cummax"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cummax = apply_dimension(
        data=input_cube,
        process=_process_cummax,
        dimension="t",
    ).compute()

    for var_name in input_cube.data_vars:
        original_max = np.max(input_cube[var_name].data, axis=0)
        cummax_total = np.max(output_cube_cummax[var_name].data, axis=0)
        assert np.all(cummax_total >= original_max)

    band_names = list(input_cube.data_vars)
    for name in band_names:
        data = input_cube[name].values
        data[:, :, 15] = np.nan
        input_cube[name] = (input_cube[name].dims, data)

    _process_cummax_with_nan = partial(
        process_registry["cummax"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cummax_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cummax_with_nan,
        dimension="t",
    ).compute()

    first_var = list(output_cube_cummax_with_nan.data_vars.values())[0]
    assert np.isnan(first_var.data[0, 0, 16])


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cummin_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    _process_cummin = partial(
        process_registry["cummin"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cummin = apply_dimension(
        data=input_cube,
        process=_process_cummin,
        dimension="t",
    ).compute()

    for var_name in input_cube.data_vars:
        original_min = np.min(input_cube[var_name].data, axis=0)
        cummin_total = np.min(output_cube_cummin[var_name].data, axis=0)
        assert np.all(cummin_total <= original_min)

    band_names = list(input_cube.data_vars)
    for name in band_names:
        data = input_cube[name].values
        data[:, :, 15] = np.nan
        input_cube[name] = (input_cube[name].dims, data)

    _process_cummin_with_nan = partial(
        process_registry["cummin"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=False,
    )

    output_cube_cummin_with_nan = apply_dimension(
        data=input_cube,
        process=_process_cummin_with_nan,
        dimension="t",
    ).compute()

    first_var = list(output_cube_cummin_with_nan.data_vars.values())[0]
    assert np.isnan(first_var.data[0, 0, 16])


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_kernel_dataset(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )
    kernel = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    output_cube = apply_kernel(data=input_cube, kernel=kernel)
    assert isinstance(output_cube, xr.Dataset)
    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}
    for var_name in output_cube.data_vars:
        assert isinstance(output_cube[var_name].data, da.Array)
    xr.testing.assert_equal(output_cube, input_cube)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_preserves_coords(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    """P0.1: apply_dimension over real dimensions preserves coordinate labels."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="numpy",
        as_dataset=True,
    )
    _process = partial(
        process_registry["add"].implementation,
        y=0,
        x=ParameterReference(from_parameter="data"),
    )
    result = apply_dimension(data=input_cube, process=_process, dimension="t")
    for var_name in input_cube.data_vars:
        np.testing.assert_array_equal(
            result[var_name].coords["t"].values,
            input_cube[var_name].coords["t"].values,
        )
        np.testing.assert_array_equal(
            result[var_name].coords["x"].values,
            input_cube[var_name].coords["x"].values,
        )


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_bands_reordering(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    """P0.2: apply_dimension over bands with reordering callback preserves correct labels."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="numpy",
        as_dataset=True,
    )

    def reverse_bands(data, axis=None, keepdims=False, **kwargs):
        return np.take(data, [3, 2, 1, 0], axis=axis)

    result = apply_dimension(data=input_cube, process=reverse_bands, dimension="bands")
    assert list(result.data_vars) == ["B08", "B04", "B03", "B02"]


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_bands_select_first(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    """P0.2: apply_dimension over bands selecting subset preserves correct labels."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="numpy",
        as_dataset=True,
    )

    def take_first(data, axis=None, keepdims=False, **kwargs):
        return np.take(data, [0], axis=axis)

    result = apply_dimension(data=input_cube, process=take_first, dimension="bands")
    assert list(result.data_vars) == ["B02"]
    assert result["B02"].attrs == input_cube["B02"].attrs
