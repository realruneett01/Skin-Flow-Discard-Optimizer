"""Slow latent drifts and injectable faults (Task 1.3).

``PressProcess`` advances the press cycle by cycle. Each latent variable wanders as
an Ornstein-Uhlenbeck process around its baseline. Faults add a ramped offset on
top, which gives the plan's "OU plus linear trend" drift when the ramp is linear.
Every cycle returns the latent ``PressState``, the sensor/actuator distortions
(``SensorEffects``) and a label dict saying which faults were active and how
strongly.

Fault kinds (names match ``Predictor.FaultClass`` where one exists):

======================  ========  ===================================================
kind                    physical  effect
======================  ========  ===================================================
die_wear                yes       die wear index rises (h_crit and tool force up)
liner_scale             yes       liner scale builds up (friction and h_crit up)
temperature_drift       yes       billet runs colder with a steeper taper
lubricant_loss          yes       friction steps up
supply_pressure_sag     yes       oil heats, supply pressure sags, ram slows, cap reading biased
sensor_gain_drift       no        cap-pressure transducer gain error
encoder_offset          no        ram position reading offset
flash_spike             yes       dummy-block flash spike early in the stroke
======================  ========  ===================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal

import numpy as np

from skinflow_discard_optimizer.config import load_config
from skinflow_discard_optimizer.sim.defect_model import PressState

Ramp = Literal["step", "linear", "exp"]

FAULT_KINDS = (
    "die_wear", "liner_scale", "temperature_drift", "lubricant_loss",
    "supply_pressure_sag", "sensor_gain_drift", "encoder_offset", "flash_spike",
)
PHYSICAL_FAULTS = {"die_wear", "liner_scale", "temperature_drift", "lubricant_loss",
                   "supply_pressure_sag", "flash_spike"}
OU_VARS = ("liner_scale_mm", "die_wear", "billet_temp_C", "liner_temp_C", "mu_base", "oil_temp_C")


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    onset_cycle: int
    severity: float = 1.0
    ramp: Ramp = "linear"
    ramp_cycles: int = 1000
    end_cycle: int | None = None    # fault removed (e.g. maintenance) from this cycle on

    def __post_init__(self):
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; expected one of {FAULT_KINDS}")
        if self.ramp not in ("step", "linear", "exp"):
            raise ValueError(f"unknown ramp {self.ramp!r}")
        if self.severity < 0 or self.ramp_cycles < 1:
            raise ValueError("severity must be >= 0 and ramp_cycles >= 1")

    def activation(self, n: int) -> float:
        """0 before onset, rising to 1 according to the ramp shape."""
        if n < self.onset_cycle or (self.end_cycle is not None and n >= self.end_cycle):
            return 0.0
        k = n - self.onset_cycle
        if self.ramp == "step":
            return 1.0
        if self.ramp == "linear":
            return min(1.0, (k + 1) / self.ramp_cycles)
        return 1.0 - float(np.exp(-(k + 1) / self.ramp_cycles))


@dataclass(frozen=True)
class SensorEffects:
    """How the measurement chain and hydraulics differ from ideal on one cycle."""

    cap_gain: float = 1.0
    cap_bias_bar: float = 0.0
    encoder_offset_mm: float = 0.0
    flash_amp_frac: float = 0.0
    flash_width_mm: float = 2.5
    speed_scale: float = 1.0
    oil_temp_C: float = 45.0
    supply_pressure_bar: float = 300.0
    supply_sag_bar: float = 0.0


def _v(node) -> float:
    return float(node["value"])


@dataclass
class HydraulicModel:
    """Oil viscosity and supply-pressure sag (placeholder model, see config/faults.yaml)."""

    nu_40: float
    b: float
    k_leak: float
    speed_loss_per_bar: float
    cap_bias_per_bar: float
    supply_nominal_bar: float
    reference_oil_temp_C: float = 45.0   # healthy operating temperature: zero sag here

    def viscosity_cSt(self, oil_temp_C):
        return self.nu_40 * np.exp(-self.b * (np.asarray(oil_temp_C) - 40.0))

    def supply_sag_bar(self, oil_temp_C):
        """Sag relative to healthy operation; the relief setting already covers normal leakage."""
        ratio = self.viscosity_cSt(self.reference_oil_temp_C) / self.viscosity_cSt(oil_temp_C)
        return np.maximum(self.k_leak * (ratio - 1.0), 0.0)


@dataclass
class FaultConfig:
    ou_theta: dict[str, float]
    ou_sigma: dict[str, float]
    oil_temp_baseline: float
    liner_zone_offsets: tuple[float, ...]
    effects: dict[str, dict[str, float]]
    hydraulic: HydraulicModel

    @classmethod
    def load(cls, supply_nominal_bar: float = 300.0) -> "FaultConfig":
        t = load_config("faults")
        h = t["hydraulic"]
        return cls(
            ou_theta={k: _v(v["theta"]) for k, v in t["ou"].items()},
            ou_sigma={k: _v(v["sigma"]) for k, v in t["ou"].items()},
            oil_temp_baseline=_v(t["baseline"]["oil_temp_C"]),
            liner_zone_offsets=tuple(t["baseline"]["liner_zone_offsets_K"]["value"]),
            effects={k: {kk: _v(vv) for kk, vv in v.items()} for k, v in t["effects"].items()},
            hydraulic=HydraulicModel(
                _v(h["nu_40_cSt"]), _v(h["viscosity_b"]), _v(h["k_leak_bar"]),
                _v(h["speed_loss_per_bar"]), _v(h["cap_bias_per_bar"]), supply_nominal_bar,
                _v(t["baseline"]["oil_temp_C"]),
            ),
        )


@dataclass
class CycleConditions:
    """Output of one ``PressProcess.step``."""

    cycle: int
    state: PressState
    effects: SensorEffects
    labels: dict[str, float]
    liner_temps_C: tuple[float, float, float, float]


@dataclass
class PressProcess:
    """Cycle-by-cycle latent state with OU wander and injected faults."""

    baseline: PressState
    faults: list[FaultSpec] = field(default_factory=list)
    config: FaultConfig = field(default_factory=FaultConfig.load)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    ou_scale: float = 1.0

    def __post_init__(self):
        self._ou = {k: 0.0 for k in OU_VARS}
        self._level_offsets: dict[str, float] = {}   # persistent shifts (die change, cold start)

    # -- events that are not faults (die change, alloy change, cold start) mutate the baseline
    def set_baseline(self, **changes) -> None:
        self.baseline = replace(self.baseline, **changes)

    def reset_ou(self, *names: str) -> None:
        for k in names or OU_VARS:
            self._ou[k] = 0.0

    def _advance_ou(self) -> None:
        for k in OU_VARS:
            th, sd = self.config.ou_theta[k], self.config.ou_sigma[k] * self.ou_scale
            self._ou[k] = (1.0 - th) * self._ou[k] + sd * self.rng.normal()

    def fault_offsets(self, n: int) -> tuple[dict[str, float], dict[str, float]]:
        """Summed effect of all faults at cycle ``n``, and per-kind activation labels."""
        offsets: dict[str, float] = {}
        labels = {f"fault_{k}": 0.0 for k in FAULT_KINDS}
        for f in self.faults:
            a = f.activation(n)
            if a <= 0.0:
                continue
            labels[f"fault_{f.kind}"] = max(labels[f"fault_{f.kind}"], f.severity * a)
            for var, per_unit in self.config.effects[f.kind].items():
                if var == "flash_width_mm":
                    offsets[var] = per_unit
                else:
                    offsets[var] = offsets.get(var, 0.0) + f.severity * a * per_unit
        return offsets, labels

    def step(self, n: int, extra_offsets: dict[str, float] | None = None) -> CycleConditions:
        self._advance_ou()
        off, labels = self.fault_offsets(n)
        for k, v in (extra_offsets or {}).items():
            off[k] = off.get(k, 0.0) + v
        b = self.baseline

        def lat(name: str, lo: float = -np.inf, hi: float = np.inf) -> float:
            val = getattr(b, name) + self._ou.get(name, 0.0) + off.get(name, 0.0)
            return float(np.clip(val, lo, hi))

        oil = self.config.oil_temp_baseline + self._ou["oil_temp_C"] + off.get("oil_temp_C", 0.0)
        hyd = self.config.hydraulic
        sag = float(hyd.supply_sag_bar(oil))
        state = replace(
            b,
            liner_scale_mm=lat("liner_scale_mm", 0.0, 3.0),
            die_wear=lat("die_wear", 0.0, 1.5),
            billet_temp_C=lat("billet_temp_C", 350.0, 560.0),
            liner_temp_C=lat("liner_temp_C", 200.0, 520.0),
            mu_base=lat("mu_base", 0.2, 0.9),
            taper_K=lat("taper_K", 0.0, 80.0),
        )
        effects = SensorEffects(
            cap_gain=1.0 + off.get("cap_gain", 0.0),
            cap_bias_bar=-hyd.cap_bias_per_bar * sag,
            encoder_offset_mm=off.get("encoder_offset_mm", 0.0),
            flash_amp_frac=off.get("flash_amp_frac", 0.0),
            flash_width_mm=off.get("flash_width_mm", 2.5),
            speed_scale=max(1.0 - hyd.speed_loss_per_bar * sag, 0.5),
            oil_temp_C=float(oil),
            supply_pressure_bar=hyd.supply_nominal_bar - sag,
            supply_sag_bar=sag,
        )
        zones = tuple(state.liner_temp_C + z for z in self.config.liner_zone_offsets)
        active = [k for k in FAULT_KINDS if labels[f"fault_{k}"] > 0]
        labels["n_active_faults"] = float(len(active))
        return CycleConditions(n, state, effects, labels, zones)  # type: ignore[arg-type]


def dominant_fault(labels: dict[str, float]) -> str:
    """Name of the most severe active fault, or 'none'."""
    best, best_v = "none", 0.0
    for k in FAULT_KINDS:
        v = labels.get(f"fault_{k}", 0.0)
        if v > best_v:
            best, best_v = k, v
    return best


def state_fields() -> list[str]:
    return [f.name for f in fields(PressState)]
