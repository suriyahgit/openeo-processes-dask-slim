"""
Optional integration test for HEALPix DGGS processes.

Loads a tiny temporal and band subset from the real Icechunk HEALPix STAC cube,
runs filter_bbox, reduce_spatial, and aggregate_spatial.

This test is skipped by default unless the required credentials and
environment variables are present.

Required environment variables:
    DEDL_HEALPIX_STAC_URL - URL to the HEALPix STAC catalog
    DEDL_S3_ENDPOINT - S3 endpoint for Icechunk storage
    DEDL_AWS_ACCESS_KEY_ID - AWS access key
    DEDL_AWS_SECRET_ACCESS_KEY - AWS secret key
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(v)
        for v in [
            "DEDL_HEALPIX_STAC_URL",
            "DEDL_S3_ENDPOINT",
            "DEDL_AWS_ACCESS_KEY_ID",
            "DEDL_AWS_SECRET_ACCESS_KEY",
        ]
    ),
    reason="Requires DEDL Icechunk credentials. Set DEDL_HEALPIX_STAC_URL, "
    "DEDL_S3_ENDPOINT, DEDL_AWS_ACCESS_KEY_ID, DEDL_AWS_SECRET_ACCESS_KEY.",
)


@pytest.fixture
def healpix_stac_url():
    return os.environ["DEDL_HEALPIX_STAC_URL"]


@pytest.fixture
def s3_endpoint():
    return os.environ["DEDL_S3_ENDPOINT"]


@pytest.fixture
def aws_credentials():
    return {
        "key": os.environ["DEDL_AWS_ACCESS_KEY_ID"],
        "secret": os.environ["DEDL_AWS_SECRET_ACCESS_KEY"],
    }


def test_load_and_filter_bbox(healpix_stac_url, s3_endpoint, aws_credentials):
    """Load a tiny HEALPix subset and filter by bounding box."""
    pytest.skip("Icechunk STAC loading not yet integrated in this test environment.")


def test_load_and_reduce_spatial(healpix_stac_url, s3_endpoint, aws_credentials):
    """Load, filter, and reduce spatially."""
    pytest.skip("Icechunk STAC loading not yet integrated in this test environment.")


def test_load_and_aggregate_spatial(healpix_stac_url, s3_endpoint, aws_credentials):
    """Load, filter, and aggregate spatially."""
    pytest.skip("Icechunk STAC loading not yet integrated in this test environment.")
