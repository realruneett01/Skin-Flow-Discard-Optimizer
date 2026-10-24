"""``skdo``: probe the host, configure the site, build models, check health, run the service.

    skdo up [-- service args]   probe -> configure -> bootstrap -> smoke test -> run   (what containers run)
    skdo probe                  show host facts and the profile derived from them
    skdo configure --press-id Press07 --namespace 3 [--endpoint opc.tcp://host:4840] [--tag name=Ident ...]
    skdo bootstrap [--force]    build or refresh models for this host and config
    skdo doctor                 verify the install; exit 1 on failure
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
    return facts, build_profile(facts)


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


def cmd_up(args) -> int:
    from skinflow_discard_optimizer.deploy import bootstrap as bs
    from skinflow_discard_optimizer.deploy.site import SITE_FILE, SiteConfig, write_site_contract

    # 1. site: honour SKDO_PRESS_ID / SKDO_OPCUA_NAMESPACE / SKDO_OPCUA_ENDPOINT when no site file exists yet.
    if not SITE_FILE.exists() and (os.environ.get("SKDO_PRESS_ID") or os.environ.get("SKDO_OPCUA_NAMESPACE")):
        cfg = SiteConfig(press_id=os.environ.get("SKDO_PRESS_ID", "Press01"),
                         namespace_index=int(os.environ.get("SKDO_OPCUA_NAMESPACE", "2")),
                         opcua_endpoint=os.environ.get("SKDO_OPCUA_ENDPOINT", ""))
        cfg.save()
        os.environ["SKDO_SIGNALS_YAML"] = str(write_site_contract(cfg))
        print(f"[up] site configured from environment: press={cfg.press_id} ns={cfg.namespace_index}", flush=True)
    # 2. host -> profile -> models
    facts, prof = _profile()
    print(f"[up] {facts.cpu_count} cores, {facts.ram_gb} GB, gpu={facts.gpu_names or 'none'} -> {', '.join(prof.reasons)}",
          flush=True)
    final = bs.bootstrap(prof)
    if not all(v == "done" for v in final.values()):
        print(f"[up] bootstrap incomplete: {final}", file=sys.stderr)
        return 1
    # 3. prove it works, then hand over to the service
    h = bs.smoke_test(3)
    print(f"[up] smoke test: {h['status']} cuts_mm={h['cuts_mm']}", flush=True)
    if args.no_run:
        return 0
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

    u = sub.add_parser("up", help="configure, bootstrap, smoke test, run")
    u.add_argument("--no-run", action="store_true", help="stop after the smoke test")
    u.add_argument("service_args", nargs=argparse.REMAINDER, help="passed to the service (e.g. --mock --mock-cycles 100)")
    u.set_defaults(fn=cmd_up)

    args = ap.parse_args(argv)
    if getattr(args, "service_args", None) and args.service_args[:1] == ["--"]:
        args.service_args = args.service_args[1:]
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
