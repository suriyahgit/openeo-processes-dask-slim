import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.mask import mask
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mask_dataset(temporal_interval, bounding_box, random_raster_data):
    data_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
        as_dataset=True,
    )

    mask_data = data_cube > 50
    output_cube = mask(data=data_cube, mask=mask_data, replacement=np.nan)

    assert set(output_cube.data_vars) == {"B02", "B03", "B04", "B08"}
    assert output_cube["B02"].isnull().any()


def test_mask_dataset_single_mask_var():
    data = xr.Dataset(
        {
            "B02": xr.DataArray(np.arange(4).reshape(2, 2), dims=["y", "x"]),
            "B03": xr.DataArray(np.arange(4).reshape(2, 2) + 10, dims=["y", "x"]),
        }
    )
    mask_data = xr.Dataset(
        {"mask": xr.DataArray([[False, True], [False, False]], dims=["y", "x"])}
    )
    out = mask(data, mask_data, replacement=-1)
    assert out["B02"][0, 1] == -1
    assert out["B03"][0, 1] == -1
    assert out["B02"][0, 0] == 0
    assert out["B03"][0, 0] == 10


def test_mask_dataset_dataarray_mask():
    data = xr.Dataset(
        {
            "B02": xr.DataArray(np.arange(4).reshape(2, 2), dims=["y", "x"]),
            "B03": xr.DataArray(np.arange(4).reshape(2, 2) + 10, dims=["y", "x"]),
        }
    )
    mask_da = xr.DataArray([[False, True], [False, False]], dims=["y", "x"])
    out = mask(data, mask_da, replacement=-1)
    assert out["B02"][0, 1] == -1
    assert out["B03"][0, 1] == -1


def test_mask_preserves_dask_laziness(temporal_interval, bounding_box):
    import dask
    import dask.array as da

    data_cube = create_fake_rastercube(
        data=np.random.rand(6, 5, 4, 2),
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03"],
        backend="dask",
        as_dataset=True,
    )
    mask_data = data_cube > 50
    output = mask(data=data_cube, mask=mask_data, replacement=np.nan)
    for var in output.data_vars.values():
        assert isinstance(var.data, dask.array.Array)
        assert len(var.data.__dask_graph__()) > 0


def test_mask_dataarray_legacy_path():
    data = xr.DataArray(np.arange(4).reshape(2, 2), dims=["y", "x"])
    mask_da = xr.DataArray(np.array([[False, True], [False, False]]), dims=["y", "x"])
    out = mask(data=data, mask=mask_da, replacement=-1)
    assert isinstance(out, xr.DataArray)
    assert out[0, 1] == -1
    assert out[0, 0] == 0
    assert out[1, 1] == 3


def test_mask_dataset_per_variable_mask():
    data = xr.Dataset(
        {
            "B02": xr.DataArray(np.arange(4).reshape(2, 2), dims=["y", "x"]),
            "B03": xr.DataArray(np.arange(4).reshape(2, 2) + 10, dims=["y", "x"]),
        }
    )
    mask_data = xr.Dataset(
        {
            "B02": xr.DataArray([[True, False], [False, False]], dims=["y", "x"]),
            "B03": xr.DataArray([[False, True], [False, False]], dims=["y", "x"]),
        }
    )
    out = mask(data, mask_data, replacement=-1)
    assert out["B02"][0, 0] == -1
    assert out["B02"][0, 1] == 1
    assert out["B03"][0, 0] == 10
    assert out["B03"][0, 1] == -1
