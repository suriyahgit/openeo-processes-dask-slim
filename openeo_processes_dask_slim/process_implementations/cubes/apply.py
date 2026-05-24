from typing import Callable, Optional, Union

import numpy as np
import odc.geo.xr
import scipy.ndimage
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.utils import (
    _capture_var_metadata,
    _detect_band_permutation,
    _restore_var_metadata,
)
from openeo_processes_dask_slim.process_implementations.data_model import RasterCube
from openeo_processes_dask_slim.process_implementations.exceptions import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
)

__all__ = ["apply", "apply_dimension", "apply_kernel"]


def apply(
    data: RasterCube, process: Callable, context: Optional[dict] = None
) -> RasterCube:
    positional_parameters = {"x": 0}
    named_parameters = {"context": context}
    result = xr.apply_ufunc(
        process,
        data,
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
        },
        keep_attrs=True,
    )
    return result


def apply_dimension(
    data: RasterCube,
    process: Callable,
    dimension: str,
    target_dimension: Optional[str] = None,
    context: Optional[dict] = None,
) -> RasterCube:
    if context is None:
        context = {}

    if dimension == "bands" and isinstance(data, xr.Dataset):
        meta = _capture_var_metadata(data)
        band_array = data.to_array(dim="bands")
        original_band_labels = band_array[dimension].values
        positional_parameters = {"data": 0}
        named_parameters = {"context": context}
        if target_dimension is None:
            target_dimension = dimension
        keepdims = target_dimension is not None
        reordered_band_array = band_array.transpose(..., dimension)
        result = xr.apply_ufunc(
            process,
            reordered_band_array,
            input_core_dims=[[dimension]],
            output_core_dims=[[dimension]],
            dask="allowed",
            kwargs={
                "positional_parameters": positional_parameters,
                "named_parameters": named_parameters,
                "axis": reordered_band_array.get_axis_num(dimension),
                "keepdims": keepdims,
                "source_transposed_axis": band_array.get_axis_num(dimension),
                "context": context,
            },
            exclude_dims={dimension},
            keep_attrs=True,
        )
        if isinstance(result, xr.DataArray) and dimension in result.dims:
            out_len = len(result[dimension])
            if out_len == len(original_band_labels):
                permuted = _detect_band_permutation(
                    result, reordered_band_array, dimension
                )
                if permuted is not None:
                    labels = permuted
                    meta["order"] = permuted
                else:
                    labels = list(original_band_labels)
                try:
                    result = result.assign_coords({dimension: labels})
                except ValueError:
                    pass
            elif out_len < len(original_band_labels):
                try:
                    result = result.assign_coords(
                        {dimension: original_band_labels[:out_len]}
                    )
                    meta["order"] = list(original_band_labels[:out_len])
                except ValueError:
                    pass
            result = result.to_dataset(dim=dimension)
        elif not isinstance(result, xr.Dataset):
            result = result.to_dataset(name="result")
        result = _restore_var_metadata(result, meta)
        if data.odc.crs is not None:
            try:
                result = odc.geo.xr.assign_crs(result, crs=data.odc.crs)
            except ValueError:
                pass
        return result

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    keepdims = False
    is_new_dim_added = target_dimension is not None
    if is_new_dim_added:
        keepdims = True

    if target_dimension is None:
        target_dimension = dimension

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    # Dataset lacks get_axis_num; compute axis from first variable
    if isinstance(data, xr.Dataset):
        sample_var = list(data.data_vars.values())[0]
        axis = sample_var.get_axis_num(dimension)
        source_axis = sample_var.get_axis_num(dimension)
        reordered_sample = list(reordered_data.data_vars.values())[0]
        reordered_axis = reordered_sample.get_axis_num(dimension)
    else:
        axis = data.get_axis_num(dimension)
        source_axis = axis
        reordered_axis = reordered_data.get_axis_num(dimension)

    result = xr.apply_ufunc(
        process,
        reordered_data,
        input_core_dims=[[dimension]],
        output_core_dims=[[dimension]],
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
            "axis": reordered_axis,
            "keepdims": keepdims,
            "source_transposed_axis": source_axis,
            "context": context,
        },
        exclude_dims={dimension},
        keep_attrs=True,
    )

    reordered_result = result.transpose(*data.dims, ...)

    if dimension in reordered_result.dims:
        result_len = len(reordered_result[dimension])
    else:
        result_len = 1

    # Case 1: target_dimension is not defined/ is source dimension
    if dimension == target_dimension:
        # dimension labels preserved
        # if the number of source dimension's values is equal to the number of computed values
        if len(reordered_data[dimension]) == result_len:
            reordered_result[dimension] = reordered_data[dimension].values
        else:
            reordered_result[dimension] = np.arange(result_len)
    elif target_dimension in reordered_result.dims:
        # source dimension is not target dimension
        # target dimension exists with a single label only
        if len(reordered_result[target_dimension]) == 1:
            reordered_result = reordered_result.drop_vars(target_dimension).squeeze(
                target_dimension
            )
            reordered_result = reordered_result.rename({dimension: target_dimension})
            reordered_result[dimension] = np.arange(result_len)
        else:
            raise Exception(
                f"Cannot rename dimension {dimension} to {target_dimension} as {target_dimension} already exists in dataset and contains more than one label: {reordered_result[target_dimension]}. See process definition. "
            )
    else:
        # source dimension is not the target dimension and the latter does not exist
        reordered_result = reordered_result.rename({dimension: target_dimension})
        reordered_result[target_dimension] = np.arange(result_len)

    if data.odc.crs is not None:
        try:
            reordered_result = odc.geo.xr.assign_crs(reordered_result, crs=data.odc.crs)
        except ValueError:
            pass

    return reordered_result


def apply_kernel(
    data: RasterCube,
    kernel: np.ndarray,
    factor: Optional[float] = 1,
    border: Union[float, str, None] = 0,
    replace_invalid: Optional[float] = 0,
) -> RasterCube:
    kernel = np.asarray(kernel)
    if any(dim % 2 == 0 for dim in kernel.shape):
        raise KernelDimensionsUneven(
            "Each dimension of the kernel must have an uneven number of elements."
        )

    def convolve(data, kernel, mode="constant", cval=0, fill_value=0):
        dims = data.openeo.spatial_dims
        convolved = lambda data: scipy.ndimage.convolve(
            data, kernel, mode=mode, cval=cval
        )

        data_masked = data.fillna(fill_value)

        if isinstance(data, xr.Dataset):
            result_vars = {}
            for var_name in data.data_vars:
                var_data = data_masked[var_name]
                result = xr.apply_ufunc(
                    convolved,
                    var_data,
                    vectorize=True,
                    dask="parallelized",
                    input_core_dims=[dims],
                    output_core_dims=[dims],
                    output_dtypes=[var_data.dtype],
                    dask_gufunc_kwargs={"allow_rechunk": True},
                ).transpose(*data[var_name].dims)
                result_vars[var_name] = result
            return xr.Dataset(
                result_vars, coords=data.coords, attrs=data.attrs
            ).transpose(*data.dims)

        dtype = data.dtype

        return xr.apply_ufunc(
            convolved,
            data_masked,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[dims],
            output_core_dims=[dims],
            output_dtypes=[dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).transpose(*data.dims)

    openeo_scipy_modes = {
        "replicate": "nearest",
        "reflect": "reflect",
        "reflect_pixel": "mirror",
        "wrap": "wrap",
    }
    if isinstance(border, int) or isinstance(border, float):
        mode = "constant"
        cval = border
    else:
        mode = openeo_scipy_modes[border]
        cval = 0

    result = convolve(data, kernel, mode, cval, replace_invalid) * factor
    result.attrs = data.attrs
    return result
