"""Live OPC-UA side of the service: subscribe to the press tags, run each cycle, publish ``Predictor.*``.

* Input: a client subscription to every tag the ``CycleAssembler`` consumes (tags from the site
  contract, ``skdo configure``). Source timestamps from the server are used when present.
* Output: this process hosts its own small OPC-UA server that exposes only the ``Predictor.*`` nodes.
  The nodes are read-only for clients, so the advisory-only guarantee does not depend on the press
  server's permissions. The press server is never written to.

Limits worth knowing: security is ``None`` by default (set ``SKDO_OPCUA_USER``/``SKDO_OPCUA_PASSWORD`` for
username auth on the client side), and sustained 1 kHz delivery depends on what the press server grants for
sampling interval and queue size; ``commission`` measures what actually arrives.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from asyncua import Client, Server, ua

from skinflow_discard_optimizer.contracts.schema import SignalContract
from skinflow_discard_optimizer.core.assembler import CycleAssembler

logger = logging.getLogger("skinflow.opcua")

# Signals the assembler consumes; everything else in the contract belongs to other modules.
REQUIRED_SIGNALS = (
    "cycle_phase", "cycle_id", "ram_position", "ram_cap_pressure", "ram_rod_pressure",
    "container_liner_temp_1", "container_liner_temp_2", "container_liner_temp_3", "container_liner_temp_4",
    "oil_temp", "billet_temp", "billet_length", "alloy_id", "die_id", "pump_supply_pressure", "pump_power",
)
TEXT = {"cycle_phase", "alloy_id", "die_id"}


def _ts(data) -> float:
    try:
        src = data.monitored_item.Value.SourceTimestamp
        if src is not None:
            return src.replace(tzinfo=src.tzinfo or timezone.utc).timestamp()
    except AttributeError:
        pass
    return time.time()


def _canonical_phase(contract: SignalContract, name: str) -> str:
    if name in (contract.signals["cycle_phase"].values or ()):
        return name
    for source in contract.phase_aliases:
        try:
            return contract.canonical_phase(name, source)
        except KeyError:
            continue
    return name


class _Handler:
    """asyncua data-change callback: turn notifications into ``(signal, t, value)`` pushes."""

    def __init__(self, by_node: dict[str, str], sink):
        self.by_node, self.sink = by_node, sink

    def datachange_notification(self, node, val, data) -> None:
        sig = self.by_node.get(node.nodeid.to_string())
        if sig is not None and val is not None:
            self.sink(sig, _ts(data), val)


def _resolve_signals(contract: SignalContract) -> dict[str, str]:
    missing = [s for s in REQUIRED_SIGNALS if s not in contract.signals]
    if missing:
        raise ValueError(f"site contract lacks signals the assembler needs: {missing}")
    return {s: contract.signals[s].tag for s in REQUIRED_SIGNALS}


async def _subscribe(client: Client, contract: SignalContract, sink) -> None:
    tags = _resolve_signals(contract)
    by_node = {client.get_node(t).nodeid.to_string(): s for s, t in tags.items()}
    handler = _Handler(by_node, sink)
    sub = await client.create_subscription(10, handler, queue_maxsize=100000)
    for sig, tag in tags.items():
        rate = contract.signals[sig].rate_hz
        interval = 1000.0 / rate if rate > 0 else 0.0
        await sub.subscribe_data_change(client.get_node(tag), queuesize=100000, sampling_interval=interval)


def make_client(endpoint: str) -> Client:
    client = Client(endpoint, timeout=10)
    user, pw = os.environ.get("SKDO_OPCUA_USER"), os.environ.get("SKDO_OPCUA_PASSWORD")
    if user:
        client.set_user(user)
        client.set_password(pw or "")
    return client


async def connect_with_retry(endpoint: str, wait_s: float = 60.0) -> Client:
    """Connect, retrying while the PLC / OPC-UA server is still starting up or briefly unreachable."""
    deadline = time.monotonic() + wait_s
    while True:
        client = make_client(endpoint)
        try:
            await client.connect()
            return client
        except (OSError, asyncio.TimeoutError, ua.UaError) as err:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"cannot reach OPC-UA server {endpoint} after {wait_s:.0f}s: {err}") from err
            logger.warning("OPC-UA server %s not reachable yet (%s); retrying", endpoint, err)
            await asyncio.sleep(2.0)


class PredictorServer:
    """Hosts the ``Predictor.*`` nodes. Clients can read and subscribe; they cannot write."""

    def __init__(self, contract: SignalContract, endpoint: str):
        self.contract, self.endpoint = contract, endpoint
        self.server = Server()
        self.nodes: dict[str, object] = {}

    async def start(self) -> None:
        s = self.server
        await s.init()
        s.set_endpoint(self.endpoint)
        s.set_server_name("Skin-Flow Discard Optimizer (advisory)")
        idx = 0
        while idx < self.contract.namespace_index:       # the contract's tags use a fixed namespace index
            idx = await s.register_namespace(f"urn:skdo:ns{idx + 1}")
        folder = await s.nodes.objects.add_folder(idx, "Predictor")
        for name, spec in self.contract.outputs.signals.items():
            init = {"float": 0.0, "bool": False}.get(spec.type, "fallback" if name.endswith("Confidence") else "none")
            node = await folder.add_variable(ua.NodeId(name, idx), name, init)   # not writable by clients
            self.nodes[name] = node
        await s.start()
        logger.info("Predictor OPC-UA server on %s (namespace %d, %d read-only nodes)",
                    self.endpoint, idx, len(self.nodes))

    async def publish(self, tags: dict) -> None:
        for name, value in tags.items():
            node = self.nodes.get(name)
            if node is None:
                continue
            spec = self.contract.outputs.signals[name]
            v = float(value) if spec.type == "float" else bool(value) if spec.type == "bool" else str(value)
            await node.write_value(v)

    async def stop(self) -> None:
        await self.server.stop()


@dataclass
class LiveRunner:
    service: object                       # StreamingOptimizerService
    contract: SignalContract
    endpoint: str
    serve: str = "opc.tcp://0.0.0.0:4841/skdo"
    results: list = field(default_factory=list)

    async def run(self, max_cycles: int | None = None, stop: asyncio.Event | None = None) -> list:
        loop = asyncio.get_running_loop()
        asm: CycleAssembler = self.service.assembler
        done: asyncio.Queue = asyncio.Queue()

        def sink(sig: str, t: float, val) -> None:
            if sig == "cycle_phase":
                val = _canonical_phase(self.contract, str(val))
            elif sig in TEXT:
                val = str(val)
            elif sig == "cycle_id":
                val = int(val)
            asm.push(sig, t, val)
            for cyc in asm.pop_completed():
                done.put_nowait(cyc)

        pub = PredictorServer(self.contract, self.serve)
        await pub.start()
        try:
            client = await connect_with_retry(self.endpoint)
            try:
                await _subscribe(client, self.contract, sink)
                logger.info("Subscribed to %d press tags on %s", len(REQUIRED_SIGNALS), self.endpoint)
                while (max_cycles is None or len(self.results) < max_cycles) and not (stop and stop.is_set()):
                    try:
                        cyc = await asyncio.wait_for(done.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    entry = await loop.run_in_executor(None, self.service.process_cycle_inputs, cyc)
                    await pub.publish(entry.published_tags)
                    self.results.append(entry)
            finally:
                await client.disconnect()
        finally:
            await pub.stop()
        return self.results


async def commission(contract: SignalContract, endpoint: str, seconds: float) -> dict:
    """Listen to the live tags for ``seconds`` and report, per signal, what actually arrives."""
    seen: dict[str, list[tuple[float, object]]] = {s: [] for s in REQUIRED_SIGNALS}

    def sink(sig, t, val):
        seen[sig].append((t, val))

    client = await connect_with_retry(endpoint, wait_s=15.0)
    try:
        await _subscribe(client, contract, sink)
        await asyncio.sleep(seconds)
    finally:
        await client.disconnect()

    report = {}
    for sig, rows in seen.items():
        spec = contract.signals[sig]
        entry: dict = {"samples": len(rows), "expected_rate_hz": spec.rate_hz}
        if rows and sig not in TEXT:
            t = np.array([r[0] for r in rows])
            v = np.array([float(r[1]) for r in rows])
            entry["observed_rate_hz"] = round(float((len(t) - 1) / max(t[-1] - t[0], 1e-9)), 2) if len(t) > 1 else 0.0
            entry["monotonic_time"] = bool(np.all(np.diff(t) >= 0))
            entry["mean"], entry["sd"] = float(v.mean()), float(v.std())
            if spec.range:
                entry["fraction_in_range"] = float(np.mean((v >= spec.range[0]) & (v <= spec.range[1])))
        report[sig] = entry
    return report
