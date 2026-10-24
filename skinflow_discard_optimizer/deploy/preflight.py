"""Deployment gate: nothing runs on a machine until this passes on that machine, with that config.

Checks, in order:

1. install  - Python, dependencies, signal contract, writable state directory, model artifacts present
              and current for this config and host profile
2. models   - a real decision cycle runs and lands inside the safety bounds
3. loopback - a simulated press is served over OPC-UA on local ports; the live path (subscribe -> assemble
              -> decide -> publish) must reproduce the direct-replay cut for the same cycle and the
              published ``Predictor.*`` nodes must be read-only
4. commission (only when an endpoint is configured) - listen to the real press tags and verify every
              required tag exists, is in range and has monotonic timestamps; report observed sample rates

The result is written to ``artifacts/preflight.json`` and is bound to a fingerprint of the config, the
contract, the artifacts and the package version; changing any of them invalidates it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from skinflow_discard_optimizer.paths import ARTIFACTS_DIR, SIGNALS_YAML

RESULT_FILE = ARTIFACTS_DIR / "preflight.json"


@dataclass
class Check:
    name: str
    status: str            # ok | warn | FAIL
    detail: str = ""


@dataclass
class Result:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
        self.checks.append(Check(name, "ok" if ok else ("warn" if warn_only else "FAIL"), detail))

    @property
    def passed(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)


def fingerprint() -> str:
    from skinflow_discard_optimizer.deploy.bootstrap import REQUIRED_ARTIFACTS, config_fingerprint

    h = hashlib.sha256(config_fingerprint().encode())
    h.update(Path(SIGNALS_YAML).read_bytes())
    for a in REQUIRED_ARTIFACTS:
        p = ARTIFACTS_DIR / a
        h.update(p.read_bytes() if p.exists() else b"missing")
    import platform

    # Bound to this machine: a preflight result copied from another computer (e.g. on a USB stick) is void.
    h.update(f"{platform.node()}|{platform.system()}|{platform.machine()}|{platform.python_version()}".encode())
    return h.hexdigest()


def is_current() -> tuple[bool, str]:
    """True when a passing preflight exists for exactly this config, contract, artifacts and version."""
    try:
        saved = json.loads(RESULT_FILE.read_text())
    except (OSError, ValueError):
        return False, "no preflight result on this machine"
    if not saved.get("passed"):
        return False, "last preflight failed"
    if saved.get("fingerprint") != fingerprint():
        return False, "config, contract or models changed since the last preflight"
    return True, f"passed {saved.get('finished')}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _loopback(contract) -> tuple[float, float, bool]:
    """Returns (live_cut_mm, replay_cut_mm, nodes_read_only)."""
    from asyncua import Client, ua

    from skinflow_discard_optimizer.service.opcua_source import LiveRunner
    from skinflow_discard_optimizer.service.run import StreamingOptimizerService
    from skinflow_discard_optimizer.service.sim_press import serve_press

    press_ep = f"opc.tcp://127.0.0.1:{_free_port()}/press"
    pred_ep = f"opc.tcp://127.0.0.1:{_free_port()}/skdo"
    press = asyncio.create_task(serve_press(contract, press_ep, cycles=1))
    live = StreamingOptimizerService()
    runner = LiveRunner(live, contract, press_ep, serve=pred_ep)
    writable = {"value": True}

    async def probe_readonly():
        await asyncio.sleep(8)       # after the Predictor server is up
        async with Client(pred_ep) as c:
            node = c.get_node(contract.outputs.signals["Predictor.ButtCutMm"].tag)
            try:
                await node.write_value(1.0)
            except Exception:  # noqa: BLE001 - any refusal is the pass condition
                writable["value"] = False

    probe = asyncio.create_task(probe_readonly())
    try:
        res = await asyncio.wait_for(runner.run(max_cycles=1), 180)
    finally:
        press.cancel()
    await probe
    ref = StreamingOptimizerService().run_mock(n_cycles=1)
    return res[0].cut_recommendation_mm, ref[0].cut_recommendation_mm, not writable["value"]


def run(endpoint: str = "", commission_s: float = 20.0, loopback: bool = True, log=print) -> Result:
    from skinflow_discard_optimizer.contracts.schema import load_contract
    from skinflow_discard_optimizer.deploy import bootstrap as bs
    from skinflow_discard_optimizer.deploy.probe import probe
    from skinflow_discard_optimizer.deploy.profile import build_profile

    r = Result()
    t0 = time.time()
    facts = probe()
    prof = build_profile(facts, sticky=True)
    r.add("host", True, f"{facts.os} {facts.arch}, {facts.cpu_count} cores, {facts.ram_gb} GB, tier={prof.tier}")

    # 1. install
    r.add("python", tuple(map(int, facts.python.split(".")[:2])) >= (3, 11), facts.python)
    try:
        contract = load_contract()
        bad = [s.tag for s in contract.signals.values() if "setpoint" in s.tag.lower()]
        r.add("signal contract", not bad, f"{SIGNALS_YAML.name}: {len(contract.signals)} inputs, "
              f"{len(contract.outputs.signals)} advisory outputs")
    except Exception as e:  # noqa: BLE001
        r.add("signal contract", False, str(e).splitlines()[0])
        return _finish(r, t0)
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS_DIR / ".w").write_text("x")
        (ARTIFACTS_DIR / ".w").unlink()
        r.add("state directory writable", True, str(ARTIFACTS_DIR))
    except OSError as e:
        r.add("state directory writable", False, str(e))
    missing = [a for a in bs.REQUIRED_ARTIFACTS if not (ARTIFACTS_DIR / a).exists()]
    r.add("model artifacts", not missing, "present" if not missing else f"missing {missing}")
    if missing:
        return _finish(r, t0)
    stale = [k for k, v in bs.status(prof).items() if v == "stale"]
    r.add("artifacts match config", not stale, "current" if not stale else f"stale: {stale}", warn_only=True)

    # 2. models
    try:
        h = bs.smoke_test(3)
        cuts = h["cuts_mm"]
        r.add("decision cycle", h["models_loaded"] and all(12.0 <= c <= 60.0 for c in cuts),
              f"cuts {cuts} mm within [12, 60]")
        r.add("latency", h["status"] != "DEGRADED",
              f"p99 {h['p99_latency_ms']:.0f} ms vs budget {prof.latency_budget_ms:.0f} ms", warn_only=True)
    except Exception as e:  # noqa: BLE001
        r.add("decision cycle", False, str(e).splitlines()[0])
        return _finish(r, t0)

    # 3. loopback through the OPC-UA stack
    if loopback:
        log("[preflight] loopback test through a simulated press (about 1 minute)...")
        try:
            live, ref, ro = asyncio.run(_loopback(contract))
            r.add("OPC-UA loopback", abs(live - ref) < 0.05,
                  f"live path {live:.2f} mm vs direct replay {ref:.2f} mm")
            r.add("Predictor nodes read-only", ro, "clients cannot write the advisory tags" if ro
                  else "a client could write the advisory tags")
        except Exception as e:  # noqa: BLE001
            r.add("OPC-UA loopback", False, f"{type(e).__name__}: {str(e)[:150]}")

    # 4. commissioning against the real press
    if endpoint:
        _commission(r, contract, endpoint, commission_s, log)
    else:
        r.add("press commissioning", True, "skipped: no OPC-UA endpoint configured", warn_only=True)
        r.checks[-1].status = "warn"
    return _finish(r, t0)


def _commission(r: Result, contract, endpoint: str, seconds: float, log) -> None:
    from skinflow_discard_optimizer.service.opcua_source import REQUIRED_SIGNALS, commission

    log(f"[preflight] commissioning: listening to {endpoint} for {seconds:.0f}s...")
    try:
        rep = asyncio.run(commission(contract, endpoint, seconds))
    except Exception as e:  # noqa: BLE001
        r.add("press commissioning", False, f"cannot read the press: {str(e)[:150]}")
        return
    dead = [s for s in REQUIRED_SIGNALS if rep[s]["samples"] == 0]
    r.add("required tags deliver data", not dead,
          "all %d tags delivered" % len(REQUIRED_SIGNALS) if not dead else f"no data from {dead}")
    oor = {s: round(v["fraction_in_range"], 3) for s, v in rep.items() if v.get("fraction_in_range", 1.0) < 0.99}
    r.add("values in contract range", not oor, "all >=99% in range" if not oor else f"out of range: {oor}")
    nonmono = [s for s, v in rep.items() if v.get("monotonic_time") is False]
    r.add("timestamps monotonic", not nonmono, "ok" if not nonmono else f"non-monotonic: {nonmono}")
    slow = {s: v["observed_rate_hz"] for s, v in rep.items()
            if v["expected_rate_hz"] >= 10 and v.get("observed_rate_hz", 0) < 0.5 * v["expected_rate_hz"]}
    r.add("sample rates", not slow, "within 50% of contract" if not slow else
          f"below half the contract rate: {slow} (press idle? server sampling limits?)", warn_only=True)
    still = [s for s in ("ram_position", "ram_cap_pressure", "ram_rod_pressure") if rep[s].get("sd", 1) == 0]
    r.add("signals vary", not still, "ram signals moved" if not still else
          f"{still} constant during the window; repeat commissioning while a billet is being extruded",
          warn_only=True)
    (ARTIFACTS_DIR / "commissioning.json").write_text(json.dumps(rep, indent=2, default=float))


def _finish(r: Result, t0: float) -> Result:
    from datetime import datetime, timezone

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps({
        "passed": r.passed,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "fingerprint": fingerprint() if all((ARTIFACTS_DIR / a).exists() for a in
                                            ("fpca.npz", "hcrit_models.npz", "ukf_params.json", "monitor_config.npz"))
        else "",
        "checks": [c.__dict__ for c in r.checks],
    }, indent=2))
    return r
