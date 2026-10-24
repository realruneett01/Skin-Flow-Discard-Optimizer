"""Deploy layer: host profile, per-press contract rendering, bootstrap cache keys."""
from dataclasses import replace

import pytest
import yaml

from skinflow_discard_optimizer.contracts.schema import SignalContract
from skinflow_discard_optimizer.deploy import bootstrap as bs
from skinflow_discard_optimizer.deploy.probe import HostFacts, probe
from skinflow_discard_optimizer.deploy.profile import build_profile
from skinflow_discard_optimizer.deploy.site import SiteConfig, render_contract, write_site_contract


def host(**kw) -> HostFacts:
    base = dict(os="Linux", arch="x86_64", python="3.12", cpu_count=4, ram_gb=16.0, in_container=False)
    return HostFacts(**{**base, **kw})


# --- profile -------------------------------------------------------------------------------------

@pytest.mark.parametrize("facts,tier", [
    (host(cpu_count=2, ram_gb=16), "edge"),
    (host(cpu_count=8, ram_gb=3), "edge"),
    (host(cpu_count=4, ram_gb=16), "standard"),
    (host(cpu_count=16, ram_gb=64), "full"),
])
def test_tier_from_host(facts, tier):
    assert build_profile(facts, env={}).tier == tier


def test_gpu_used_only_with_cuda_torch_and_not_on_edge():
    assert build_profile(host(cpu_count=16, ram_gb=64, torch_cuda=True), env={}).device == "cuda"
    assert build_profile(host(cpu_count=16, ram_gb=64, gpu_names=("A100",)), env={}).device == "cpu"
    assert build_profile(host(cpu_count=1, ram_gb=2, torch_cuda=True), env={}).device == "cpu"


def test_workers_leave_a_core_free_and_respect_overrides():
    p = build_profile(host(cpu_count=4), env={})
    assert 1 <= p.workers <= 3
    assert build_profile(host(), env={"SKDO_WORKERS": "2", "SKDO_TIER": "full"}).workers == 2
    assert build_profile(host(), env={"SKDO_TIER": "full"}).dataset_scale == 1.0
    with pytest.raises(ValueError):
        build_profile(host(), env={"SKDO_DEVICE": "tpu"})


def test_probe_reports_something_sane():
    h = probe()
    assert h.cpu_count >= 1 and h.ram_gb >= 0 and h.disk_free_gb > 0


# --- site contract -------------------------------------------------------------------------------

def test_default_site_reproduces_shipped_contract():
    doc = render_contract(SiteConfig())
    from skinflow_discard_optimizer.contracts.schema import load_contract
    shipped = load_contract()
    assert {k: v.tag for k, v in shipped.signals.items()} == {k: v["tag"] for k, v in doc["signals"].items()}


def test_press_id_namespace_and_override_applied():
    doc = render_contract(SiteConfig(press_id="Press07", namespace_index=3,
                                     tag_overrides={"ram_cap_pressure": "PLC.Main.CapP"}))
    c = SignalContract.model_validate(doc)
    assert c.signals["ram_cap_pressure"].tag == "ns=3;s=PLC.Main.CapP"
    assert c.signals["ram_position"].tag == "ns=3;s=Press07.Ram.Position"
    assert all(s.tag.startswith("ns=3;s=Predictor.") for s in c.outputs.signals.values())


def test_outputs_cannot_be_overridden_and_bad_ids_rejected():
    with pytest.raises(ValueError, match="unknown input signals"):
        render_contract(SiteConfig(tag_overrides={"Predictor.ButtCutMm": "Plant.Cut"}))
    with pytest.raises(ValueError):
        SiteConfig(press_id="Press 07").validate()
    with pytest.raises(ValueError):
        SiteConfig(tag_overrides={"ram_position": "Bad-Id"}).validate()


def test_duplicate_tag_override_fails_contract_validation():
    with pytest.raises(ValueError):
        render_contract(SiteConfig(tag_overrides={"ram_position": "Press01.Ram.CapPressure"}))


def test_write_site_contract_roundtrip(tmp_path):
    dest = write_site_contract(SiteConfig(press_id="P9"), tmp_path / "signals.yaml")
    assert yaml.safe_load(dest.read_text())["signals"]["cycle_id"]["tag"] == "ns=2;s=P9.State.CycleId"


# --- bootstrap cache keys ------------------------------------------------------------------------

def test_keys_ignore_workers_and_device_but_not_fit_sizes():
    base = build_profile(host(cpu_count=16, ram_gb=64), env={})
    k = bs.stage_keys(base)
    assert k == bs.stage_keys(replace(base, workers=1, device="cpu" if base.device == "cuda" else "cuda"))
    changed = bs.stage_keys(replace(base, n_train=base.n_train + 1))
    assert changed["dataset"] == k["dataset"] and changed["decision"] != k["decision"]
    # a dataset change invalidates every later stage
    assert all(v != k[n] for n, v in bs.stage_keys(replace(base, dataset_scale=0.5)).items())


def test_config_change_invalidates_keys(monkeypatch, tmp_path):
    p = build_profile(host(), env={})
    before = bs.stage_keys(p)
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "x.yaml").write_text("a: 1")
    monkeypatch.setattr(bs, "CONFIG_DIR", cfg)
    assert bs.stage_keys(p) != before
