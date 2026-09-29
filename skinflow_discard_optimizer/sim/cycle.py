"""One simulated press cycle: latent state + faults -> stroke + per-cycle signals + truth.

``simulate_cycle`` returns a flat row of scalars (what the press reports once per
cycle, plus every ground-truth label) and the 1 kHz stroke. The row stores every
parameter of the stroke and its own ``stroke_seed``, so ``regenerate_stroke(row)``
rebuilds the identical 1 kHz stream later. The dataset therefore never has to store
billions of raw samples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from skinflow_discard_optimizer.config import load_config, value
from skinflow_discard_optimizer.sim.defect_model import CycleTruth, DefectModel
from skinflow_discard_optimizer.sim.faults import FAULT_KINDS, CycleConditions, dominant_fault
from skinflow_discard_optimizer.sim.force_model import (
    Alloy,
    Press,
    StrokeData,
    StrokeSpec,
    force_breakdown,
    simulate_stroke,
)

DCTO_PHASES = ("decompression", "container_shift_open", "shear_stroke", "die_slide",
               "billet_load", "container_shift_close", "rapid_advance")

SPEC_FIELDS = ("L0_mm", "extrusion_ratio", "ram_speed_mm_s", "T_front_C", "taper_K",
               "heating_K", "heating_length_mm", "sigma_scale", "mu", "F_tool_N",
               "h_onset_mm", "upturn_amp_N", "lam_mm", "drop_fraction", "fill_length_mm",
               "entry_amplitude_frac", "entry_length_mm", "accel_tau_s", "h_end_mm")


@dataclass(frozen=True)
class CycleContext:
    """Things that change only at events: alloy, die, curve shape."""

    alloy: str = "AA6063"
    die_id: str = "D-101"
    extrusion_ratio: float = 40.0
    shape: str = "upturn"


@dataclass(frozen=True)
class Coupling:
    phase_nominal: dict[str, float]
    phase_sd: dict[str, float]
    shear_s_per_mm: float
    shear_reference_mm: float

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "Coupling":
        t = load_config("coupling")
        ph = t["dead_cycle_phases"]
        return cls(
            {k: float(v["nominal_s"]["value"]) for k, v in ph.items()},
            {k: float(v["sd_s"]["value"]) for k, v in ph.items()},
            float(value(t, "shear.seconds_per_mm_discard")),
            float(value(t, "shear.reference_cut_mm")),
        )

    def shear_time_s(self, h_cut_mm) -> np.ndarray:
        """Expected shear-stroke duration for a given discard thickness."""
        return self.phase_nominal["shear_stroke"] + self.shear_s_per_mm * (
            np.asarray(h_cut_mm) - self.shear_reference_mm)


def _flash_force(spec: StrokeSpec, press: Press, amp_frac: float, width_mm: float,
                 rng: np.random.Generator):
    if amp_frac <= 0.0:
        return None, np.nan
    f_start = float(force_breakdown(np.array([10.0]), spec, press).total_N[0])
    x_f = float(rng.uniform(2.0, 0.04 * spec.L0_mm))
    amp = amp_frac * f_start

    def extra(x):
        return amp * np.exp(-0.5 * ((x - x_f) / width_mm) ** 2)

    return extra, x_f


def pump_energy_kwh(d: StrokeData, press: Press) -> float:
    """Hydraulic input energy of the stroke: integral of p_cap * Q / eta (true signals)."""
    p_cap_MPa = (d.force_true_N + press.rod_back_pressure_bar / 10 * press.rod_area_mm2) / press.cap_area_mm2
    q_mm3_s = d.v_mm_s * press.cap_area_mm2
    power_W = p_cap_MPa * q_mm3_s * 1e-3 / press.pump_overall_efficiency
    dt = np.diff(d.t_s, prepend=d.t_s[0])
    return float(np.sum(power_W * dt) / 3.6e6)


def simulate_cycle(cond: CycleConditions, model: DefectModel, ctx: CycleContext,
                   stroke_seed: int, process_rng: np.random.Generator,
                   static_cut_mm: float | None = None,
                   full_stroke: bool = True) -> tuple[dict, StrokeData | None]:
    """Simulate one cycle. ``process_rng`` draws truth; ``stroke_seed`` drives sensor noise.

    With ``full_stroke=False`` the 1 kHz stroke is skipped: energy and duration come
    from a noise-free 50 Hz pass (integration difference well under 0.1%), and the
    returned stroke is None. The row is identical either way apart from that, and
    ``regenerate_stroke(row)`` gives the full stroke on demand.
    """
    press, eff, s = model.press, cond.effects, cond.state
    # Per-billet variation: saw-cut length and ram-speed control scatter (config/press.yaml).
    length = s.billet_length_mm + float(process_rng.normal(0.0, press.billet_length_sd_mm))
    speed = (s.ram_speed_mm_s + float(process_rng.normal(0.0, press.ram_speed_sd_mm_s))) * eff.speed_scale
    s = type(s)(**{**asdict(s), "ram_speed_mm_s": max(speed, 1.0), "billet_length_mm": length})
    truth: CycleTruth = model.sample_truth(s, process_rng, shape=ctx.shape)  # type: ignore[arg-type]
    spec = model.stroke_for(s, truth, process_rng, extrusion_ratio=ctx.extrusion_ratio)

    stroke_rng = np.random.default_rng(stroke_seed)
    extra, flash_x = _flash_force(spec, press, eff.flash_amp_frac, eff.flash_width_mm, stroke_rng)
    if full_stroke:
        d = simulate_stroke(spec, press, stroke_rng, extra_force_N=extra, cap_gain=eff.cap_gain,
                            cap_bias_bar=eff.cap_bias_bar, encoder_offset_mm=eff.encoder_offset_mm)
    else:
        d = simulate_stroke(spec, press, stroke_rng, fs_hz=50.0, noise=False, extra_force_N=extra)

    static = model.p.static_cut_mm if static_cut_mm is None else static_cut_mm
    oracle = float(model.oracle_cut(np.array(truth.h_crit_mm)))
    cpl = Coupling.load()
    nz = lambda sd: float(process_rng.normal(0.0, sd))  # noqa: E731
    tn = press.temp_noise_C

    row: dict = {
        "cycle": cond.cycle,
        "stroke_seed": int(stroke_seed),
        "alloy_id": ctx.alloy,
        "die_id": ctx.die_id,
        "shape": ctx.shape,
        # --- what the press reports (measured, per cycle)
        "billet_length_mm": s.billet_length_mm,
        "billet_temp_C": s.billet_temp_C + nz(tn),
        **{f"container_liner_temp_{i + 1}": z + nz(tn) for i, z in enumerate(cond.liner_temps_C)},
        "oil_temp_C": eff.oil_temp_C + nz(0.3),
        "pump_supply_pressure_min_bar": eff.supply_pressure_bar - abs(nz(1.0)),
        "pump_energy_kwh": pump_energy_kwh(d, press),
        "stroke_duration_s": float(d.t_s[-1]),
        **{f"phase_{p}_s": cpl.phase_nominal[p] + nz(cpl.phase_sd[p]) for p in DCTO_PHASES},
        # --- ground truth
        "h_crit_mm": truth.h_crit_mm,
        "h_crit_mean_mm": truth.h_crit_mean_mm,
        "oracle_cut_mm": oracle,
        "static_cut_mm": static,
        "static_defect_prob": float(model.defect_probability(static, truth.h_crit_mm)),
        "static_cost_eur": float(model.expected_cost(static, truth.h_crit_mm)),
        "oracle_cost_eur": float(model.expected_cost(oracle, truth.h_crit_mm)),
        **{f"spec_{k}": float(getattr(spec, k)) for k in SPEC_FIELDS},
        "true_liner_scale_mm": s.liner_scale_mm,
        "true_die_wear": s.die_wear,
        "true_billet_temp_C": s.billet_temp_C,
        "true_liner_temp_C": s.liner_temp_C,
        "true_mu_base": s.mu_base,
        "true_dT_K": s.dT_K,
        "eff_cap_gain": eff.cap_gain,
        "eff_cap_bias_bar": eff.cap_bias_bar,
        "eff_encoder_offset_mm": eff.encoder_offset_mm,
        "eff_flash_amp_frac": eff.flash_amp_frac,
        "eff_flash_width_mm": eff.flash_width_mm,
        "eff_flash_x_mm": flash_x,
        "eff_speed_scale": eff.speed_scale,
        "eff_supply_sag_bar": eff.supply_sag_bar,
        **{k: float(v) for k, v in cond.labels.items()},
        "fault_class": dominant_fault(cond.labels),
    }
    # The shear time depends on the cut actually made; under static operation that is `static`.
    row["phase_shear_stroke_s"] = float(cpl.shear_time_s(static)) + nz(cpl.phase_sd["shear_stroke"])
    return row, (d if full_stroke else None)


def spec_from_row(row: dict, alloy: Alloy | None = None) -> StrokeSpec:
    alloy = alloy or Alloy.from_config(row["alloy_id"])
    kw = {k: float(row[f"spec_{k}"]) for k in SPEC_FIELDS}
    return StrokeSpec(alloy=alloy, shape=row["shape"], **kw)


def regenerate_stroke(row: dict, press: Press | None = None) -> StrokeData:
    """Rebuild the exact 1 kHz stroke of a dataset row from its stored parameters and seed."""
    press = press or Press.load()
    spec = spec_from_row(row)
    rng = np.random.default_rng(int(row["stroke_seed"]))
    extra = None
    if row["eff_flash_amp_frac"] > 0:
        # consume the same draw simulate_cycle made for the flash position
        extra, _ = _flash_force(spec, press, row["eff_flash_amp_frac"], row["eff_flash_width_mm"], rng)
    return simulate_stroke(spec, press, rng, extra_force_N=extra, cap_gain=row["eff_cap_gain"],
                           cap_bias_bar=row["eff_cap_bias_bar"],
                           encoder_offset_mm=row["eff_encoder_offset_mm"])


LABEL_COLUMNS = tuple(f"fault_{k}" for k in FAULT_KINDS) + ("n_active_faults", "fault_class")
