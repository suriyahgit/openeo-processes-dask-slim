import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask_slim.process_implementations.cubes.dggs import (
    apply_neighborhood_dggs,
    mask_polygon_dggs,
)
from tests.mockdata import create_fake_healpix_cube


@pytest.fixture
def cube_n2():
    return create_fake_healpix_cube(nside=2, n_times=3, backend="numpy")


@pytest.fixture
def polygon_mask():
    return {
        "type": "Polygon",
        "coordinates": [[[-45, -45], [-45, 45], [45, 45], [45, -45], [-45, -45]]],
    }


class TestApplyNeighborhoodDggs:
    def test_requires_k_or_radius(self, cube_n2):
        def dummy(data, **kwargs):
            return np.mean(data)

        with pytest.raises(ValueError):
            apply_neighborhood_dggs(cube_n2, dummy)

    def test_k_1_returns_same_shape(self, cube_n2):
        def _mean(data, **kwargs):
            return np.mean(data)

        result = apply_neighborhood_dggs(cube_n2, _mean, k=1)
        assert len(result.healpix_index) == len(cube_n2.healpix_index)

    def test_k_1_preserves_temporal(self, cube_n2):
        def _mean(data, **kwargs):
            return np.mean(data)

        result = apply_neighborhood_dggs(cube_n2, _mean, k=1)
        assert result.dims["t"] == cube_n2.dims["t"]

    def test_k_1_preserves_bands(self, cube_n2):
        def _mean(data, **kwargs):
            return np.mean(data)

        result = apply_neighborhood_dggs(cube_n2, _mean, k=1)
        for var in cube_n2.data_vars:
            assert var in result.data_vars

    def test_k_1_values_different(self, cube_n2):
        def _mean(data, **kwargs):
            return np.mean(data)

        result = apply_neighborhood_dggs(cube_n2, _mean, k=1)
        assert not np.allclose(result["band_0"].values, cube_n2["band_0"].values)

    def test_include_center_false(self, cube_n2):
        def _sum(data, **kwargs):
            return np.sum(data)

        result = apply_neighborhood_dggs(cube_n2, _sum, k=1, include_center=False)
        assert len(result.healpix_index) == len(cube_n2.healpix_index)

    def test_k_2_works(self, cube_n2):
        def _mean(data, **kwargs):
            return np.mean(data)

        result = apply_neighborhood_dggs(cube_n2, _mean, k=2)
        assert len(result.healpix_index) == len(cube_n2.healpix_index)

    def test_k_greater_than_4_rejected(self, cube_n2):
        def dummy(data, **kwargs):
            return np.mean(data)

        with pytest.raises(ValueError):
            apply_neighborhood_dggs(cube_n2, dummy, k=5)

    def test_radius_not_implemented(self, cube_n2):
        def dummy(data, **kwargs):
            return np.mean(data)

        with pytest.raises(NotImplementedError):
            apply_neighborhood_dggs(cube_n2, dummy, radius=10.0)

    def test_with_context(self, cube_n2):
        def _weighted_mean(data, context=None, **kwargs):
            if context is not None and "weights" in context:
                return np.average(data, weights=context["weights"])
            return np.mean(data)

        result = apply_neighborhood_dggs(
            cube_n2, _weighted_mean, k=1, context={"weights": [1, 1, 1, 1, 1, 1, 1, 1]}
        )
        assert len(result.healpix_index) == len(cube_n2.healpix_index)


class TestMaskPolygonDggs:
    def test_masks_outside_polygon(self, cube_n2, polygon_mask):
        result = mask_polygon_dggs(cube_n2, polygon_mask)
        masked_band = result["band_0"].values
        assert np.any(np.isnan(masked_band))

    def test_inside_true(self, cube_n2, polygon_mask):
        result = mask_polygon_dggs(cube_n2, polygon_mask, inside=True)
        masked_band = result["band_0"].values
        assert np.any(np.isnan(masked_band))

    def test_custom_replacement(self, cube_n2, polygon_mask):
        result = mask_polygon_dggs(cube_n2, polygon_mask, replacement=-999)
        masked_band = result["band_0"].values
        assert np.any(masked_band == -999)

    def test_preserves_temporal(self, cube_n2, polygon_mask):
        result = mask_polygon_dggs(cube_n2, polygon_mask)
        assert result.dims["t"] == cube_n2.dims["t"]

    def test_preserves_healpix_dim(self, cube_n2, polygon_mask):
        result = mask_polygon_dggs(cube_n2, polygon_mask)
        assert result.dims["healpix_index"] == cube_n2.dims["healpix_index"]

    def test_preserves_attrs(self, cube_n2, polygon_mask):
        result = mask_polygon_dggs(cube_n2, polygon_mask)
        assert result.attrs["healpix_nside"] == 2
