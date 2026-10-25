"""``skdo``: probe the host, configure the site, build models, check health, run the service.

    skdo up [-- service args]   probe -> configure -> bootstrap -> smoke test -> run   (what containers run)
    skdo probe                  show host facts and the profile derived from them
    skdo configure --press-id Press07 --namespace 3 [--endpoint opc.tcp://host:4840] [--tag name=Ident ...]
    skdo bootstrap [--force]    build or refresh models for this host and config
    skdo doctor                 verify the install; exit 1 on failure
    skdo preflight [--endpoint URL]   the deployment gate: install, models, OPC-UA loopback, press commissioning
    skdo simulate-press         serve a simulated press over OPC-UA (no hardware needed)
    skdo bundle OUT --target cp312-win_amd64 ...   build a plug-and-run USB folder
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _site_env() -> None:
    # Must run before anything imports skinflow_discard_optimizer.paths.
    from skinflow_discard_optimizer.deploy.site import apply_site_env

    apply_site_env()


def _profile():
    from skinflow_discard_optimizer.deploy.probe import probe
    from skinflow_discard_optimizer.deploy.profile import build_profile

    facts = probe()
    return facts, build_profile(facts, sticky=True)


def cmd_probe(args) -> int:
    facts, prof = _profile()
    print(json.dumps({"host": facts.to_dict(), "profile": prof.to_dict()}, indent=2))
    return 0


def cmd_configure(args) -> int:
    from skinflow_discard_optimizer.deploy.site import SITE_FILE, SiteConfig, render_contract, write_site_contract

    base = SiteConfig.load()
    overrides = dict(base.tag_overrides)
    for item in args.tag or []:
        name, _, ident = item.partition("=")
        if not ident:
            print(f"--tag expects name=Identifier, got {item!r}", file=sys.stderr)
            return 2
        overrides[name] = ident
    cfg = SiteConfig(
        press_id=args.press_id or base.press_id,
        namespace_index=args.namespace if args.namespace is not None else base.namespace_index,
        opcua_endpoint=args.endpoint if args.endpoint is not None else base.opcua_endpoint,
        tag_overrides=overrides,
    )
    try:
        cfg.validate()
        render_contract(cfg)  # validate before anything is written, so a bad flag never leaves a broken site file
    except ValueError as e:
        print(f"invalid site config: {e}", file=sys.stderr)
        return 2
    cfg.save()
    dest = write_site_contract(cfg)
    print(f"wrote {SITE_FILE} and {dest}")
    return 0


def cmd_bootstrap(args) -> int:
    from skinflow_discard_optimizer.deploy import bootstrap as bs

    facts, prof = _profile()
    print(f"[bootstrap] {', '.join(prof.reasons)}", flush=True)
    if args.status:
        print(json.dumps(bs.status(prof), indent=2))
        return 0
    final = bs.bootstrap(prof, force=args.force)
    print(json.dumps(final, indent=2))
    return 0 if all(v == "done" for v in final.values()) else 1


def _check(results: list[tuple[str, str, str]], name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    results.append((name, "ok" if ok else ("warn" if warn_only else "FAIL"), detail))


def cmd_doctor(args) -> int:
    results: list[tuple[str, str, str]] = []
    _check(results, "python", sys.version_info >= (3, 11), sys.version.split()[0])
    for mod in ("numpy", "scipy", "pandas", "pyarrow", "pydantic", "yaml"):
        try:
            m = __import__(mod)
            _check(results, f"import {mod}", True, getattr(m, "__version__", ""))
        except ImportError as e:
            _check(results, f"import {mod}", False, str(e))

    from skinflow_discard_optimizer.deploy import bootstrap as bs
    from skinflow_discard_optimizer.paths import ARTIFACTS_DIR, SIGNALS_YAML

    try:
        from skinflow_discard_optimizer.contracts.schema import load_contract_file

        c = load_contract_file(SIGNALS_YAML)
        setpoints = [s.tag for s in c.signals.values() if "setpoint" in s.tag.lower()]
        _check(results, "signal contract", not setpoints,
               f"{SIGNALS_YAML} ({len(c.signals)} inputs, {len(c.outputs.signals)} Predictor.* outputs)")
    except Exception as e:  # noqa: BLE001 - report any contract problem as a failed check
        _check(results, "signal contract", False, str(e).splitlines()[0])

    try:
        from skinflow_discard_optimizer.config import iter_leaves, load_config

        names = ("press", "alloys", "dies", "economics", "defect", "decision", "coupling", "faults")
        leaves = [(n, k, v) for n in names for k, v in iter_leaves(load_config(n))]
        ph = sum(1 for _, _, v in leaves if v.get("status") == "PLACEHOLDER")
        _check(results, "config provenance", True, f"{len(leaves)} parameters, {ph} still PLACEHOLDER", warn_only=True)
        if ph:
            results[-1] = ("config provenance", "warn", f"{ph}/{len(leaves)} parameters are PLACEHOLDER "
                           "(calibrate on this press before trusting economics)")
    except Exception as e:  # noqa: BLE001
        _check(results, "config provenance", False, str(e).splitlines()[0])

    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        probe_file = ARTIFACTS_DIR / ".write_test"
        probe_file.write_text("x")
        probe_file.unlink()
        _check(results, "artifacts dir writable", True, str(ARTIFACTS_DIR))
    except OSError as e:
        _check(results, "artifacts dir writable", False, str(e))

    facts, prof = _profile()
    _check(results, "host", True, f"{facts.cpu_count} cores, {facts.ram_gb} GB, tier={prof.tier}, device={prof.device}")
    missing = [a for a in bs.REQUIRED_ARTIFACTS if not (ARTIFACTS_DIR / a).exists()]
    _check(results, "model artifacts", not missing,
           "all present" if not missing else f"missing {missing} (run `skdo bootstrap`)")
    if not missing:
        stale = [k for k, v in bs.status(prof).items() if v == "stale"]
        _check(results, "artifacts current", not stale,
               "match config + profile" if not stale else f"stale stages {stale} (run `skdo bootstrap`)",
               warn_only=True)
    if not args.quick and not missing:
        try:
            h = bs.smoke_test(3)
            _check(results, "smoke test", h["models_loaded"] and h["status"] != "ERROR",
                   f"status={h['status']} cuts_mm={h['cuts_mm']} p99={h['p99_latency_ms']:.0f}ms "
                   f"(budget {prof.latency_budget_ms:.0f}ms)")
            if h["status"] == "DEGRADED" and results[-1][1] == "ok":
                results[-1] = (results[-1][0], "warn", results[-1][2] + " - latency budget exceeded on this host")
        except Exception as e:  # noqa: BLE001
            _check(results, "smoke test", False, str(e).splitlines()[0])

    width = max(len(n) for n, _, _ in results)
    for name, state, detail in results:
        print(f"{state:>4}  {name:<{width}}  {detail}")
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


def _endpoint(args) -> str:
    from skinflow_discard_optimizer.deploy.site import SiteConfig

    return getattr(args, "endpoint", None) or os.environ.get("SKDO_OPCUA_ENDPOINT") or SiteConfig.load().opcua_endpoint


def cmd_preflight(args) -> int:
    from skinflow_discard_optimizer.deploy import preflight

    res = preflight.run(endpoint=_endpoint(args), commission_s=args.seconds, loopback=not args.no_loopback)
    width = max(len(c.name) for c in res.checks)
    for c in res.checks:
        print(f"{c.status:>4}  {c.name:<{width}}  {c.detail}")
    print("\nPREFLIGHT " + ("PASSED" if res.passed else "FAILED - the service will not start on this machine"))
    return 0 if res.passed else 1


def cmd_simulate_press(args) -> int:
    import asyncio

    from skinflow_discard_optimizer.contracts.schema import load_contract
    from skinflow_discard_optimizer.service.sim_press import serve_press

    print(f"simulated press on {args.listen} (Ctrl+C to stop)")
    try:
        asyncio.run(serve_press(load_contract(), args.listen, cycles=args.cycles, pace=args.pace))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_verify_bundle(args) -> int:
    from pathlib import Path

    from skinflow_discard_optimizer.deploy.bundle import verify_manifest

    problems = verify_manifest(Path(args.root))
    if problems:
        print("BUNDLE INTEGRITY CHECK FAILED - refusing to start:", file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        return 1
    print("[verify] bundle intact")
    return 0


def cmd_gen_cert(args) -> int:
    import asyncio
    from pathlib import Path

    from skinflow_discard_optimizer.service.opcua_security import generate_certificate

    cert, key = asyncio.run(generate_certificate(Path(args.out), args.name, args.host, f"urn:{args.host}:skdo:{args.name}"))
    print(f"wrote {cert} and {key}\nserver: SKDO_SERVE_CERT={cert} SKDO_SERVE_KEY={key}\n"
          f"client of the press: SKDO_OPCUA_CERT={cert} SKDO_OPCUA_KEY={key} (and trust it on the press server)")
    return 0


def cmd_bundle(args) -> int:
    from pathlib import Path

    from skinflow_discard_optimizer.deploy.bundle import build_bundle

    build_bundle(Path(args.out), args.target, with_artifacts=not args.no_artifacts, embed_python=args.embed_python)
    return 0


def cmd_up(args) -> int:
    from skinflow_discard_optimizer.deploy import bootstrap as bs
    from skinflow_discard_optimizer.deploy import preflight
    from skinflow_discard_optimizer.deploy.site import SITE_FILE, SiteConfig, write_site_contract

    # 1. site: honour SKDO_PRESS_ID / SKDO_OPCUA_NAMESPACE / SKDO_OPCUA_ENDPOINT when no site file exists yet.
    if not SITE_FILE.exists() and (os.environ.get("SKDO_PRESS_ID") or os.environ.get("SKDO_OPCUA_NAMESPACE")):
        cfg = SiteConfig(press_id=os.environ.get("SKDO_PRESS_ID", "Press01"),
                         namespace_index=int(os.environ.get("SKDO_OPCUA_NAMESPACE", "2")),
                         opcua_endpoint=os.environ.get("SKDO_OPCUA_ENDPOINT", ""))
        cfg.save()
        os.environ["SKDO_SIGNALS_YAML"] = str(write_site_contract(cfg))
        print(f"[up] site configured from environment: press={cfg.press_id} ns={cfg.namespace_index}", flush=True)
    # 2. host -> profile -> models (prebuilt artifacts on a USB stick are used as they are)
    facts, prof = _profile()
    print(f"[up] {facts.cpu_count} cores, {facts.ram_gb} GB, gpu={facts.gpu_names or 'none'} -> {', '.join(prof.reasons)}",
          flush=True)
    final = bs.bootstrap(prof)
    if not all(v == "done" for v in final.values()):
        print(f"[up] bootstrap incomplete: {final}", file=sys.stderr)
        return 1
    # 3. the gate: must have passed on this machine with this exact config and these models
    ok, why = preflight.is_current()
    if not ok:
        print(f"[up] {why}; running preflight", flush=True)
        res = preflight.run(endpoint=_endpoint(args), loopback=not args.skip_loopback)
        for c in res.checks:
            print(f"[preflight] {c.status:>4}  {c.name}: {c.detail}", flush=True)
        if not res.passed:
            print("[up] preflight FAILED - not starting. Fix the FAIL lines above (see `skdo preflight`).", file=sys.stderr)
            return 1
    else:
        print(f"[up] preflight {why}", flush=True)
    if args.no_run:
        return 0
    # 4. run: live press if an endpoint is configured and no explicit mock/replay was requested
    endpoint = _endpoint(args)
    if endpoint and not {"--mock", "--replay"} & set(args.service_args):
        import asyncio
        import logging

        from skinflow_discard_optimizer.contracts.schema import load_contract
        from skinflow_discard_optimizer.service.opcua_source import LiveRunner
        from skinflow_discard_optimizer.service.run import ServiceConfig, StreamingOptimizerService

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        svc = StreamingOptimizerService(ServiceConfig(latency_budget_ms=prof.latency_budget_ms))
        runner = LiveRunner(svc, load_contract(), endpoint, serve=args.serve)
        try:
            asyncio.run(runner.run())
        except KeyboardInterrupt:
            pass
        return 0
    if not {"--mock", "--replay"} & set(args.service_args):
        print("[up] nothing to read from: no OPC-UA endpoint is configured. Set SKDO_OPCUA_ENDPOINT, pass --endpoint, run "
              "`skdo configure --endpoint ...`, or demo with `skdo up -- --mock --mock-cycles 20`.", file=sys.stderr)
        return 2
    from skinflow_discard_optimizer.service import run as service

    sys.argv = ["skinflow-service", "--budget-ms", str(prof.latency_budget_ms), *args.service_args]
    service.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    _site_env()
    ap = argparse.ArgumentParser(prog="skdo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="show host facts and derived profile").set_defaults(fn=cmd_probe)

    c = sub.add_parser("configure", help="write site config and per-press signal contract")
    c.add_argument("--press-id")
    c.add_argument("--namespace", type=int)
    c.add_argument("--endpoint")
    c.add_argument("--tag", action="append", metavar="NAME=IDENT", help="override one signal's string identifier")
    c.set_defaults(fn=cmd_configure)

    b = sub.add_parser("bootstrap", help="build or refresh models")
    b.add_argument("--force", action="store_true")
    b.add_argument("--status", action="store_true", help="only report stage status")
    b.set_defaults(fn=cmd_bootstrap)

    d = sub.add_parser("doctor", help="verify the install")
    d.add_argument("--quick", action="store_true", help="skip the smoke test (for container health checks)")
    d.set_defaults(fn=cmd_doctor)

    pf = sub.add_parser("preflight", help="deployment gate; exit 1 if this machine must not run the service")
    pf.add_argument("--endpoint", help="press OPC-UA endpoint (default: site config)")
    pf.add_argument("--seconds", type=float, default=20.0, help="commissioning listen time")
    pf.add_argument("--no-loopback", action="store_true", help="skip the simulated-press OPC-UA loopback")
    pf.set_defaults(fn=cmd_preflight)

    sp = sub.add_parser("simulate-press", help="serve a simulated press over OPC-UA")
    sp.add_argument("--listen", default="opc.tcp://127.0.0.1:4842/press")
    sp.add_argument("--cycles", type=int, default=0, help="cycles to play once; 0 = repeat forever")
    sp.add_argument("--pace", type=float, default=0.0, help="wall seconds per simulated second (0 = fast)")
    sp.set_defaults(fn=cmd_simulate_press)

    gc = sub.add_parser("gen-cert", help="create a self-signed OPC-UA certificate and key")
    gc.add_argument("--out", default="site/certs")
    gc.add_argument("--name", default="skdo")
    gc.add_argument("--host", default="localhost", help="host name the certificate is for")
    gc.set_defaults(fn=cmd_gen_cert)

    bd = sub.add_parser("bundle", help="build a plug-and-run USB folder")
    bd.add_argument("out")
    bd.add_argument("--target", action="append", required=True, metavar="cpXYZ-PLATFORM",
                    help="e.g. cp312-win_amd64, cp312-linux_x86_64, cp312-linux_aarch64, cp312-macos_arm64")
    bd.add_argument("--no-artifacts", action="store_true", help="do not copy prebuilt models (built on first run instead)")
    bd.add_argument("--embed-python", metavar="ZIP", help="Windows embeddable Python zip to bundle (no install needed)")
    bd.set_defaults(fn=cmd_bundle)

    vb = sub.add_parser("verify-bundle", help="check a USB bundle against its MANIFEST.sha256")
    vb.add_argument("root")
    vb.set_defaults(fn=cmd_verify_bundle)

    u = sub.add_parser("up", help="configure, bootstrap, preflight gate, run")
    u.add_argument("--endpoint", help="press OPC-UA endpoint (default: site config)")
    u.add_argument("--serve", default="opc.tcp://127.0.0.1:4841/skdo", help="where the read-only Predictor.* server listens (non-loopback needs SKDO_SERVE_CERT/KEY)")
    u.add_argument("--skip-loopback", action="store_true")
    u.add_argument("--no-run", action="store_true", help="stop after the preflight gate")
    u.add_argument("service_args", nargs=argparse.REMAINDER, help="passed to the service (e.g. --mock --mock-cycles 100)")
    u.set_defaults(fn=cmd_up)

    args = ap.parse_args(argv)
    if getattr(args, "service_args", None) and args.service_args[:1] == ["--"]:
        args.service_args = args.service_args[1:]
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
