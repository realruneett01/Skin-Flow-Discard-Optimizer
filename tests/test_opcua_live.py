"""Live OPC-UA path against the bundled simulated press. Needs built artifacts; slow (~1 min)."""
import asyncio

import pytest

from skinflow_discard_optimizer.deploy.bootstrap import REQUIRED_ARTIFACTS
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not all((ARTIFACTS_DIR / a).exists() for a in REQUIRED_ARTIFACTS), reason="models not built")
def test_loopback_matches_replay_and_outputs_read_only():
    from skinflow_discard_optimizer.contracts.schema import load_contract
    from skinflow_discard_optimizer.deploy.preflight import _loopback

    live, ref, read_only = asyncio.run(_loopback(load_contract()))
    assert abs(live - ref) < 0.05
    assert read_only
