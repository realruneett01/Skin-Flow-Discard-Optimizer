"""OPC-UA input robustness: whatever a press server sends, the sink must not raise; and the link must recover."""
import asyncio
import socket

import numpy as np
import pytest

from skinflow_discard_optimizer.contracts.schema import load_contract
from skinflow_discard_optimizer.core.assembler import CycleAssembler
from skinflow_discard_optimizer.deploy.bootstrap import REQUIRED_ARTIFACTS
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR
from skinflow_discard_optimizer.service.opcua_source import REQUIRED_SIGNALS, make_sink

CONTRACT = load_contract()
GARBAGE = [None, "", "abc", "nan", b"\x00\xff", True, False, [], [1, 2], {}, {"a": 1}, (1,), object(), 10**400,
           -10**400, float("nan"), float("inf"), float("-inf"), 0, 1, -1, 1e308, np.float32(3.0), np.int64(5),
           np.array([1.0, 2.0]), "extrusion", "dwell", 12.5, "billet_load"]


@pytest.mark.parametrize("seed", range(300))
def test_sink_never_raises_on_arbitrary_values(seed):
    rng = np.random.default_rng(800_000 + seed)
    asm = CycleAssembler()
    cycles, bad = [], {}
    sink = make_sink(CONTRACT, asm, cycles.append, bad)
    t = 0.0
    for _ in range(int(rng.integers(30, 300))):
        sig = str(rng.choice(REQUIRED_SIGNALS))
        val = GARBAGE[int(rng.integers(0, len(GARBAGE)))]
        t += float(rng.choice([0.001, 0.01, 0.0, -1.0]))
        sink(sig, t if rng.integers(0, 20) else float("nan"), val)
    sink("cycle_phase", t + 1, "dwell")
    assert all(v > 0 for v in bad.values())
    for c in cycles:
        assert np.isfinite(c.x_mm).all()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
@pytest.mark.skipif(not all((ARTIFACTS_DIR / a).exists() for a in REQUIRED_ARTIFACTS), reason="models not built")
def test_link_loss_is_recovered():
    """Kill the press server mid-run, bring it back, and the service must resume and finish the next cycle."""
    from skinflow_discard_optimizer.service.opcua_source import LiveRunner
    from skinflow_discard_optimizer.service.run import ServiceConfig, StreamingOptimizerService
    from skinflow_discard_optimizer.service.sim_press import serve_press

    async def go():
        port = free_port()
        ep = f"opc.tcp://127.0.0.1:{port}/press"
        svc = StreamingOptimizerService(ServiceConfig(latency_budget_ms=1e9))
        runner = LiveRunner(svc, CONTRACT, ep, serve=f"opc.tcp://127.0.0.1:{free_port()}/skdo", stale_s=3.0)
        run_task = asyncio.create_task(runner.run(max_cycles=2))
        first = asyncio.create_task(serve_press(CONTRACT, ep, cycles=1, startup_wait_s=2.0))
        while not runner.results:                      # first cycle published
            await asyncio.sleep(0.5)
            assert not run_task.done()
        first.cancel()                                 # the press goes away
        await asyncio.gather(first, return_exceptions=True)
        await asyncio.sleep(6.0)                       # long enough to be declared stale and attempt reconnects
        second = asyncio.create_task(serve_press(CONTRACT, ep, cycles=1, startup_wait_s=2.0))
        try:
            await asyncio.wait_for(run_task, 150)
        finally:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
        return runner.results

    results = asyncio.run(go())
    assert len(results) == 2 and all(12.0 <= r.cut_recommendation_mm <= 60.0 for r in results)
