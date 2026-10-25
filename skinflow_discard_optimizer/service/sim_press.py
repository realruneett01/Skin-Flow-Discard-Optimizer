"""A simulated press PLC: an OPC-UA server that plays digital-twin cycles through the site's tags.

Lets a machine be checked end to end (subscription, assembly, decision, publication) with no press attached:
``skdo simulate-press`` in one terminal, ``skdo up --opcua opc.tcp://localhost:4840/press`` in another.
Samples carry the recorded source timestamps, so playback can run faster than real time without
distorting the physics (the force model uses time derivatives).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from asyncua import Server, ua

from skinflow_discard_optimizer.contracts.schema import SignalContract
from skinflow_discard_optimizer.service.opcua_source import REQUIRED_SIGNALS, TEXT
from skinflow_discard_optimizer.service.replay import cycle_to_records
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario

logger = logging.getLogger("skinflow.simpress")


async def serve_press(contract: SignalContract, endpoint: str, scenario: str = "healthy_baseline",
                      cycles: int = 3, startup_wait_s: float = 3.0, pace: float = 0.0) -> None:
    """Serve ``cycles`` simulated cycles once, then keep serving until cancelled.

    ``cycles`` > 0 plays that many cycles once; 0 repeats forever. ``pace`` is seconds of wall-clock per simulated
    second (0 = as fast as possible).
    """
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)
    idx = 0
    while idx < contract.namespace_index:
        idx = await server.register_namespace(f"urn:simpress:ns{idx + 1}")
    folder = await server.nodes.objects.add_folder(idx, "Press")
    nodes = {}
    for sig in REQUIRED_SIGNALS:
        spec = contract.signals[sig]
        ident = spec.tag.split(";s=", 1)[1]
        # Start every tag at a valid value (like a real PLC); a 0.0 default is out of range for temperatures.
        lo, hi = spec.range if spec.range else (0.0, 0.0)
        init = ("idle" if sig == "cycle_phase" else "") if sig in TEXT else (
            int(lo) if spec.type == "int" else float(lo + 0.5 * (hi - lo)))
        nodes[sig] = await folder.add_variable(ua.NodeId(ident, idx), ident, init)
    async with server:
        logger.info("Simulated press on %s; first cycle in %.0fs", endpoint, startup_wait_s)
        await asyncio.sleep(startup_wait_s)           # let the client subscribe before data flows
        epoch = datetime.now(timezone.utc)
        t0 = 0.0
        sc = Scenario.load(scenario).with_cycles(cycles if cycles > 0 else 200)
        while True:                                   # cycles == 0: keep producing, like a real press
            for row, stroke in run_scenario(sc, keep_strokes=True):
                rec, t0 = cycle_to_records(row, stroke, t0)
                last_t = rec["t"].iloc[0]
                for t, sig, val, text in rec[["t", "signal", "value", "text"]].itertuples(index=False):
                    if sig not in nodes:
                        continue
                    if pace > 0:
                        await asyncio.sleep(max(t - last_t, 0.0) * pace)
                    last_t = t
                    v = text if sig in TEXT else (int(val) if contract.signals[sig].type == "int" else float(val))
                    dv = ua.DataValue(ua.Variant(v), SourceTimestamp=epoch + timedelta(seconds=float(t)))
                    await nodes[sig].write_value(dv)
                    if int(t * 1000) % 50 == 0:
                        await asyncio.sleep(0)        # yield so the server can flush notifications
                if cycles <= 0:
                    await asyncio.sleep(0.5)          # pace whole cycles when running forever
            if cycles > 0:
                break
        logger.info("Simulated cycles finished; still serving")
        await asyncio.Event().wait()
