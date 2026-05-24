try:
    import dask
except ImportError:
    dask = None

import numpy as np
import xarray as xr
from xarray.core.duck_array_ops import isnull as xr_isnull


def _has_dask():
    return dask is not None


def _is_dask_array(arr):
    return _has_dask() and isinstance(arr, dask.array.Array)


def isnull(data):
    if _is_dask_array(data):
        return dask.array.map_blocks(xr_isnull, data)
    else:
        return xr_isnull(data)


def notnull(data):
    return ~isnull(data)


def ensure_raster_cube(data, process_name=None):
    if not isinstance(data, xr.Dataset):
        msg = (
            f"RasterCube must be an xr.Dataset"
            f"{' in ' + process_name if process_name else ''}"
            f", got {type(data).__name__}. "
            f"DataArray inputs are no longer supported."
        )
        raise TypeError(msg)


def _detect_band_permutation(result_da, input_da, dimension):
    """
    Detect permutation of band labels when a callback reorders along dimension.

    After apply_ufunc with output_core_dims=[[dimension]], xarray preserves
    the input dimension coordinate values even when the callback reorders data
    along that axis. This function samples the first valid element of each
    output position and matches it to the corresponding input position.

    Parameters
    ----------
    result_da : xr.DataArray
        The DataArray returned by apply_ufunc (with dimension in dims).
    input_da : xr.DataArray
        The input DataArray before apply_ufunc (same dims as result_da).
    dimension : str
        The dimension name along which the callback was applied.

    Returns
    -------
    list or None
        Permuted labels if a unique 1:1 mapping was detected, or None if
        detection is ambiguous (identical band data, count mismatch, etc.)
    """
    n_out = len(result_da[dimension])
    n_in = len(input_da[dimension])
    if n_out != n_in or n_out == 0:
        return None

    # Sample the first element along all non-band dimensions
    idx = {d: 0 for d in result_da.dims if d != dimension}
    if not idx:
        # Only the dimension itself exists; compare full 1D arrays
        out_data = result_da.values
        in_data = input_da.values
    else:
        out_data = result_da.isel(idx).values
        in_data = input_da.isel(idx).values

    # Flatten to 1D if needed
    out_data = out_data.ravel()
    in_data = in_data.ravel()

    if len(out_data) != len(in_data):
        return None

    permutation = [-1] * n_out
    used = [False] * n_in

    for out_pos in range(n_out):
        for in_pos in range(n_in):
            if used[in_pos]:
                continue
            match = np.allclose(
                out_data[out_pos], in_data[in_pos], atol=0, rtol=0, equal_nan=True
            )
            if match:
                permutation[out_pos] = in_pos
                used[in_pos] = True
                break

    if -1 in permutation or not all(used):
        return None

    original_labels = list(input_da[dimension].values)
    return [original_labels[p] for p in permutation]


def _capture_var_metadata(dataset):
    """Capture per-variable attrs and variable order before to_array()."""
    return {
        "attrs": {v: dataset[v].attrs for v in dataset.data_vars},
        "order": list(dataset.data_vars),
    }


def _restore_var_metadata(result, metadata):
    """Restore per-variable attrs and variable order after to_dataset()."""
    for v, attrs in metadata["attrs"].items():
        if v in result.data_vars:
            result[v].attrs = attrs
    try:
        result = result[metadata["order"]]
    except KeyError:
        pass
    return result
