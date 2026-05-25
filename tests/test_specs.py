import json
from pathlib import Path

specs_path = Path(__file__).parent.parent / "openeo_processes_dask_slim/specs/openeo-processes"


def test_specs():
    from openeo_processes_dask_slim.specs import reduce_dimension

    assert reduce_dimension == json.load(open(specs_path / "reduce_dimension.json"))

    # Make sure we don't overwrite any builtins
    from openeo_processes_dask_slim.specs import _all

    assert _all == json.load(open(specs_path / "all.json"))
