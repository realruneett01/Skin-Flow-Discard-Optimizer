"""Randomised checks of the deploy layer: site contracts, host profiles, bundle manifests, security env, assembler input."""
import string

import numpy as np
import pytest
import yaml

from skinflow_discard_optimizer.contracts.schema import SignalContract, load_contract
from skinflow_discard_optimizer.core.assembler import CycleAssembler
from skinflow_discard_optimizer.deploy import bootstrap as bs
from skinflow_discard_optimizer.deploy import bundle
from skinflow_discard_optimizer.deploy.probe import HostFacts
from skinflow_discard_optimizer.deploy.profile import _TIERS, build_profile
from skinflow_discard_optimizer.deploy.site import SiteConfig, render_contract
from skinflow_discard_optimizer.service.opcua_security import ServerSecurity

CONTRACT = load_contract()
INPUTS = list(CONTRACT.signals)
ID_CHARS = string.ascii_letters + string.digits + "_"
TAG_CHARS = ID_CHARS + "."
BAD_CHARS = " -;=/\\:'\"\n\t%$#@!"


def rand_str(rng, chars, lo, hi):
    return "".join(rng.choice(list(chars), int(rng.integers(lo, hi + 1))))


# ---------------------------------------------------------------------------------------- site contract
@pytest.mark.parametrize("seed", range(300))
def test_valid_site_configs_render_valid_contracts(seed):
    rng = np.random.default_rng(seed)
    overrides = {}
    for name in rng.choice(INPUTS, int(rng.integers(0, 5)), replace=False):
        overrides[str(name)] = "Site." + rand_str(rng, TAG_CHARS, 3, 20) + f".{name}"   # unique per signal
    cfg = SiteConfig(press_id=rand_str(rng, ID_CHARS, 1, 16), namespace_index=int(rng.integers(0, 65536)),
                     tag_overrides=overrides)
    doc = render_contract(cfg)
    c = SignalContract.model_validate(yaml.safe_load(yaml.safe_dump(doc)))     # survives a YAML round trip
    ns = f"ns={cfg.namespace_index};s="
    assert all(s.tag.startswith(ns) for s in c.signals.values())
    assert all(s.tag.startswith(ns + "Predictor.") for s in c.outputs.signals.values())
    for name, ident in overrides.items():
        assert c.signals[name].tag == ns + ident
    others = [n for n in INPUTS if n not in overrides]
    assert all(c.signals[n].tag.startswith(ns + cfg.press_id + ".") for n in others)
    assert len({s.tag for s in c.signals.values()} | {s.tag for s in c.outputs.signals.values()}) == len(c.signals) + len(c.outputs.signals)


@pytest.mark.parametrize("seed", range(150))
def test_invalid_site_configs_fail_cleanly(seed):
    rng = np.random.default_rng(50_000 + seed)
    kind = seed % 5
    if kind == 0:
        cfg = SiteConfig(press_id=rand_str(rng, ID_CHARS, 0, 3) + str(rng.choice(list(BAD_CHARS))) + "x")
    elif kind == 1:
        cfg = SiteConfig(namespace_index=int(rng.choice([-1, 65536, 10**6])))
    elif kind == 2:
        cfg = SiteConfig(tag_overrides={"nonexistent_" + rand_str(rng, ID_CHARS, 1, 5): "A.B"})
    elif kind == 3:
        cfg = SiteConfig(tag_overrides={str(rng.choice(INPUTS)): rand_str(rng, ID_CHARS, 1, 5) + str(rng.choice(list(BAD_CHARS)))})
    else:                                           # duplicate tags between two signals
        a, b = (str(x) for x in rng.choice(INPUTS, 2, replace=False))
        cfg = SiteConfig(tag_overrides={a: "Dup.Tag", b: "Dup.Tag"})
    with pytest.raises(ValueError):
        cfg.validate()
        render_contract(cfg)


# ---------------------------------------------------------------------------------------- host profile
@pytest.mark.parametrize("seed", range(300))
def test_profile_invariants_for_any_host(seed):
    rng = np.random.default_rng(100_000 + seed)
    cpus = int(rng.choice([1, 2, 3, 4, 6, 8, 16, 32, 64, 128]))
    ram = float(rng.choice([0.0, 0.5, 1, 2, 3.9, 4, 8, 15.9, 16, 64, 512]))
    h = HostFacts("Linux", "x86_64", "3.12", cpus, ram, bool(rng.integers(0, 2)),
                  gpu_names=("GPU",) if rng.integers(0, 2) else (), torch_cuda=bool(rng.integers(0, 2)), disk_free_gb=10.0)
    p = build_profile(h, env={})
    assert p.tier in _TIERS
    assert 1 <= p.workers <= max(1, cpus)
    assert 0 < p.dataset_scale <= 1.0 and p.feature_stride >= 1
    assert p.n_train > 0 and p.n_cal > 0 and p.fit_stride >= 1
    assert p.device in ("cpu", "cuda")
    if p.device == "cuda":
        assert h.torch_cuda and p.tier != "edge"
    if cpus <= 2 or ram < 4:
        assert p.tier == "edge"
    if cpus >= 8 and ram >= 16:
        assert p.tier == "full"
    forced = build_profile(h, env={"SKDO_TIER": ["edge", "standard", "full"][seed % 3]})
    assert forced.tier == ["edge", "standard", "full"][seed % 3]


# ---------------------------------------------------------------------------------------- bootstrap keys
@pytest.mark.parametrize("seed", range(100))
def test_stage_keys_are_deterministic_and_chained(seed):
    rng = np.random.default_rng(150_000 + seed)
    h = HostFacts("Linux", "x86_64", "3.12", int(rng.integers(1, 64)), float(rng.integers(1, 256)), False)
    p = build_profile(h, env={})
    k1, k2 = bs.stage_keys(p), bs.stage_keys(p)
    assert k1 == k2 and len(set(k1.values())) == len(k1)
    from dataclasses import replace
    assert bs.stage_keys(replace(p, workers=p.workers + 1)) == k1                 # speed knobs do not matter
    names = list(k1)
    field = ["dataset_scale", "fpca_n", "feature_stride", "ukf_n_tune", "n_train"][seed % 5]
    changed = bs.stage_keys(replace(p, **{field: getattr(p, field) * 2}))
    first = next(i for i, n in enumerate(names) if changed[n] != k1[n])
    assert all(changed[n] != k1[n] for n in names[first:])                         # downstream stages all invalidated
    assert all(changed[n] == k1[n] for n in names[:first])


# ---------------------------------------------------------------------------------------- bundle manifest
@pytest.mark.parametrize("seed", range(150))
def test_manifest_detects_exactly_the_changes_made(tmp_path, seed):
    rng = np.random.default_rng(200_000 + seed)
    files = {}
    for i in range(int(rng.integers(2, 12))):
        sub = str(rng.choice(["app", "lib", "app/config", "lib/pkg"]))
        path = tmp_path / sub / f"f{i}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rng.bytes(int(rng.integers(0, 300))))
        files[path] = path.relative_to(tmp_path).as_posix()
    state = tmp_path / "app" / "artifacts" / "s.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{}")
    bundle.write_manifest(tmp_path)
    assert bundle.verify_manifest(tmp_path) == []
    state.write_text('{"x": 1}')                                              # mutable state may change freely
    assert bundle.verify_manifest(tmp_path) == []
    victim = list(files)[int(rng.integers(0, len(files)))]
    action = seed % 3
    if action == 0:
        victim.write_bytes(victim.read_bytes() + b"x")
        expect = f"modified: {files[victim]}"
    elif action == 1:
        victim.unlink()
        expect = f"missing: {files[victim]}"
    else:
        extra = tmp_path / "lib" / "added.py"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("x")
        expect = "unexpected file: lib/added.py"
    assert expect in bundle.verify_manifest(tmp_path)


@pytest.mark.parametrize("seed", range(100))
def test_parse_target_accepts_only_valid_targets(seed):
    rng = np.random.default_rng(300_000 + seed)
    valid = f"cp3{int(rng.integers(10, 14))}-{rng.choice(list(bundle.PLATFORMS))}"
    assert bundle.parse_target(valid)[0] == valid.split("-")[0]
    for bad in (valid.replace("cp", "py"), valid.split("-")[0], valid + "x", "-" + valid.split("-")[1],
                "cpx12-win_amd64", rand_str(rng, ID_CHARS, 1, 12)):
        with pytest.raises(ValueError):
            bundle.parse_target(bad)


# ---------------------------------------------------------------------------------------- security env
@pytest.mark.parametrize("seed", range(100))
def test_server_security_env_combinations(seed):
    rng = np.random.default_rng(400_000 + seed)
    keys = ["SKDO_SERVE_CERT", "SKDO_SERVE_KEY", "SKDO_SERVE_USER", "SKDO_SERVE_PASSWORD"]
    env = {k: "v" for k in keys if rng.integers(0, 2)}
    pair_ok = (("SKDO_SERVE_CERT" in env) == ("SKDO_SERVE_KEY" in env)
               and ("SKDO_SERVE_USER" in env) == ("SKDO_SERVE_PASSWORD" in env))
    if pair_ok:
        sec = ServerSecurity.from_env(env)
        assert sec.encrypted == ("SKDO_SERVE_CERT" in env)
        routable = "opc.tcp://192.168.1.2:4841/x"
        if sec.encrypted:
            sec.check_bind(routable)
        else:
            with pytest.raises(PermissionError):
                sec.check_bind(routable)
    else:
        with pytest.raises(ValueError):
            ServerSecurity.from_env(env)


# ---------------------------------------------------------------------------------------- assembler robustness
NUM_SIGNALS = ["ram_position", "ram_cap_pressure", "ram_rod_pressure", "container_liner_temp_1", "oil_temp",
               "pump_supply_pressure", "pump_power"]
PHASES = list(CONTRACT.signals["cycle_phase"].values) + ["bogus", ""]


@pytest.mark.parametrize("seed", range(150))
def test_assembler_never_crashes_on_garbage(seed):
    rng = np.random.default_rng(500_000 + seed)
    asm = CycleAssembler()
    t = 0.0
    for _ in range(int(rng.integers(20, 400))):
        t += float(rng.choice([0.001, 0.0, -0.5, 1.0, np.nan, np.inf]))
        r = rng.integers(0, 10)
        if r < 2:
            asm.push("cycle_phase", t, str(rng.choice(PHASES)))
        elif r == 2:
            asm.push(str(rng.choice(["alloy_id", "die_id"])), t, rand_str(rng, ID_CHARS, 0, 8))
        elif r == 3:
            asm.push(str(rng.choice(["billet_temp", "billet_length", "cycle_id"])), t,
                     float(rng.choice([450.0, np.nan, np.inf, -1e9, 1e9])))
        else:
            asm.push(str(rng.choice(NUM_SIGNALS)), t, float(rng.choice([0.0, 1.0, 250.0, np.nan, np.inf, -np.inf, 1e12])))
    asm.push("cycle_phase", t + 1, "dwell")
    for cyc in asm.pop_completed():
        assert len(cyc.t_s) == len(cyc.x_mm) == len(cyc.p_cap_bar) == len(cyc.p_rod_bar)
        assert np.isfinite(cyc.x_mm).all()           # out-of-range / non-finite samples are rejected before assembly
    assert all(v >= 0 for v in asm.rejected.values())
