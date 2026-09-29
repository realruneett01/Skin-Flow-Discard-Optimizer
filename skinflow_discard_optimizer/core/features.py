"""Per-cycle feature row, identical at training time and live time (Task 2.3).

Both paths build a ``CycleInputs`` and call the same ``FeatureExtractor.extract``:

* training/batch: ``inputs_from_row(row, stroke)``, a dataset row plus its regenerated
  1 kHz stroke;
* live/replay: ``core.assembler.CycleAssembler``, which turns OPC-UA tag samples
  into ``CycleInputs``.

Features:

* FPCA scores and SPE of the registered curve (``fpca_*``, ``fpca_spe``)
* fitted ``theta = [sigma_scale, mu, F_tool]`` from a least-squares fit of the
  physics baseline, plus the fit's residual RMS
* ``upturn_h_mm``: where the end-of-stroke force departs from the baseline trend,
  using only data down to ``h_stop_mm`` (NaN if no departure is seen)
* cross-project: oil temperature, minimum supply pressure during the stroke,
  dead-cycle step durations, shear time, pump energy per cycle
* context: alloy, die, billet temperature and length, liner temperatures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from skinflow_discard_optimizer.config import load_config
from skinflow_discard_optimizer.core.functional import FPCA, RegistrationGrid, registered_curve
from skinflow_discard_optimizer.core.preprocess import resample_to_position, robust_noise_sd
from skinflow_discard_optimizer.sim.cycle import DCTO_PHASES
from skinflow_discard_optimizer.sim.force_model import (
    KELVIN,
    Alloy,
    Press,
    StrokeData,
    feltham_strain_rate,
    flow_stress_MPa,
    pressures_to_force,
)


@lru_cache(maxsize=1)
def die_registry() -> dict[str, float]:
    return {k: float(v["extrusion_ratio"]["value"]) for k, v in load_config("dies").items()}


@dataclass
class CycleInputs:
    """Everything the feature layer needs about one cycle, whatever the source."""

    cycle_id: int
    alloy_id: str
    die_id: str
    t_s: np.ndarray
    x_mm: np.ndarray
    p_cap_bar: np.ndarray
    p_rod_bar: np.ndarray
    billet_temp_C: float
    billet_length_mm: float
    liner_temps_C: tuple[float, float, float, float]
    oil_temp_C: float
    supply_pressure_min_bar: float
    pump_energy_kwh: float
    phase_durations_s: dict[str, float] = field(default_factory=dict)
    extrusion_ratio: float | None = None

    def ratio(self) -> float:
        return self.extrusion_ratio if self.extrusion_ratio is not None else die_registry()[self.die_id]


def inputs_from_row(row: dict, stroke: StrokeData) -> CycleInputs:
    """Batch path: a dataset row plus its regenerated stroke."""
    return CycleInputs(
        cycle_id=int(row["cycle"]),
        alloy_id=row["alloy_id"],
        die_id=row["die_id"],
        t_s=stroke.t_s,
        x_mm=stroke.x_mm,
        p_cap_bar=stroke.p_cap_bar,
        p_rod_bar=stroke.p_rod_bar,
        billet_temp_C=float(row["billet_temp_C"]),
        billet_length_mm=float(row["billet_length_mm"]),
        liner_temps_C=tuple(float(row[f"container_liner_temp_{i}"]) for i in range(1, 5)),  # type: ignore[arg-type]
        oil_temp_C=float(row["oil_temp_C"]),
        supply_pressure_min_bar=float(row["pump_supply_pressure_min_bar"]),
        pump_energy_kwh=float(row["pump_energy_kwh"]),
        phase_durations_s={p: float(row[f"phase_{p}_s"]) for p in DCTO_PHASES},
        extrusion_ratio=float(row["extrusion_ratio"]) if "extrusion_ratio" in row else None,
    )


@dataclass
class ThetaFit:
    sigma_scale: float
    mu: float
    F_tool_N: float
    resid_rms_N: float


def fit_theta(x_mm: np.ndarray, force_N: np.ndarray, L0_mm: float, R: float,
              *args, **kwargs) -> ThetaFit:
    """Least-squares fit of the plan's baseline force model (no upturn term).

    With flow stress fixed by the measured temperature and speed, the model
    ``F = s*Ac*sigma*lnR + (s*mu)*Ac*sigma*4*(L0-x)/Db + F_tool`` is linear in
    ``(s, s*mu, F_tool)``, so the fit is exact and fast. The temperature profile uses
    the measured front temperature with the nominal taper and heating from
    config/press.yaml, since those are not measured per billet.
    """
    v_mm_s: float = args[0] if len(args) > 0 else kwargs["v_mm_s"]
    T_front_C: float = args[1] if len(args) > 1 else kwargs["T_front_C"]
    alloy: Alloy = args[2] if len(args) > 2 else kwargs["alloy"]
    press: Press = args[3] if len(args) > 3 else kwargs["press"]

    Db = press.container_bore_mm
    Ac = press.container_area_mm2
    T = (T_front_C - press.taper_K * x_mm / L0_mm
         + press.deformation_heating_K * (1 - np.exp(-x_mm / press.heating_length_mm)) + KELVIN)
    sigma = flow_stress_MPa(feltham_strain_rate(v_mm_s, Db, R), T, alloy)
    A = np.column_stack([Ac * sigma * np.log(R), Ac * sigma * 4 * (L0_mm - x_mm) / Db, np.ones_like(x_mm)])
    coef, *_ = np.linalg.lstsq(A, force_N, rcond=None)
    s, smu, ft = coef
    resid = force_N - A @ coef
    return ThetaFit(float(s), float(smu / s) if s != 0 else float("nan"), float(ft), float(np.sqrt(np.mean(resid**2))))


def _find_upturn_run(out: np.ndarray, h: np.ndarray, run_mm: float, n: int) -> int | None:
    end = len(out) - 1 - n // 2
    if end < 1 or not out[end]:
        return None
    i = end
    while i > 0 and out[i - 1]:
        i -= 1
    if (h[i] - h[end]) < run_mm:
        return None
    return i


def _fit_exponential_decay(h_run: np.ndarray, r_run: np.ndarray, sd: float, ref_amplitude_N: float) -> tuple[float, float]:
    pos = r_run > 2 * sd
    if pos.sum() < 5:
        return float("nan"), float("nan")
    hh, rr = h_run[pos], r_run[pos]
    slope, icpt = np.polyfit(hh, np.log(rr), 1, w=np.sqrt(rr))
    if slope >= 0:
        return float("nan"), float("nan")
    lam = -1.0 / slope
    return float(lam * (icpt - np.log(ref_amplitude_N))), float(lam)


def upturn_location(x_mm: np.ndarray, force_N: np.ndarray, L0_mm: float, h_stop_mm: float,
                    *args, **kwargs) -> tuple[float, float]:
    """End-of-stroke onset thickness and decay length, or ``(nan, nan)`` if none is seen.

    ``F_up = a*exp(-h/lam)`` has no sharp start, so "where it leaves the noise"
    would depend on the noise level. The onset is therefore *defined* as the
    thickness where the upturn reaches ``ref_amplitude_N`` (the same convention the
    simulator uses), which makes it comparable across presses and noise levels.
    """
    ref_amplitude_N: float = args[0] if len(args) > 0 else kwargs["ref_amplitude_N"]
    k_sigma: float = args[1] if len(args) > 1 else kwargs.get("k_sigma", 5.0)
    dx_mm: float = args[2] if len(args) > 2 else kwargs.get("dx_mm", 0.5)
    run_mm: float = args[3] if len(args) > 3 else kwargs.get("run_mm", 3.0)

    keep = (L0_mm - x_mm) >= h_stop_mm
    c = resample_to_position(x_mm[keep], force_N[keep], dx_mm, x_min=max(L0_mm - 130.0, 0.0))
    h = L0_mm - c.x_mm
    ref = (h >= 60) & (h <= 110)
    if ref.sum() < 20:
        return float("nan"), float("nan")
    coef = np.polyfit(c.x_mm[ref], c.force_N[ref], 1)
    r = c.force_N - np.polyval(coef, c.x_mm)
    sd = robust_noise_sd(c.force_N[ref])
    n = max(int(run_mm / dx_mm), 1)
    rs = np.convolve(r, np.ones(n) / n, mode="same")
    thr = k_sigma * sd / np.sqrt(n)
    out = (np.abs(rs) > thr) & (h < 60)

    i = _find_upturn_run(out, h, run_mm, n)
    if i is None:
        return float("nan"), float("nan")
    run = slice(i, len(out))
    if np.median(r[run]) < 0:
        return float(h[i]), float("nan")
    return _fit_exponential_decay(h[run], r[run], sd, ref_amplitude_N)


class FeatureExtractor:
    """Turns ``CycleInputs`` into one feature row. The same object serves batch and live."""

    def __init__(self, press: Press | None = None, fpca: FPCA | None = None,
                 grid: RegistrationGrid | None = None, h_stop_mm: float = 12.0):
        self.press = press or Press.load()
        self.fpca = fpca
        self.grid = grid or (fpca.grid if fpca is not None else RegistrationGrid())
        self.h_stop_mm = h_stop_mm
        self._alloys: dict[str, Alloy] = {}

    def alloy(self, name: str) -> Alloy:
        if name not in self._alloys:
            self._alloys[name] = Alloy.from_config(name)
        return self._alloys[name]

    def extract(self, ci: CycleInputs) -> dict:
        press = self.press
        L0 = press.upset_length(ci.billet_length_mm)
        R = ci.ratio()
        force = pressures_to_force(ci.p_cap_bar, ci.p_rod_bar, press)
        h = L0 - ci.x_mm
        seen = h >= self.h_stop_mm            # causal horizon: nothing past the stop point
        x, f, t = ci.x_mm[seen], force[seen], ci.t_s[seen]

        out: dict = {"cycle": ci.cycle_id, "alloy_id": ci.alloy_id, "die_id": ci.die_id,
                     "extrusion_ratio": R, "L0_mm": L0}

        reg = registered_curve(x, f, L0, self.grid)
        if self.fpca is not None:
            p = self.fpca.project(reg)
            out.update({f"fpca_{k + 1}": float(v) for k, v in enumerate(p.scores)})
            out["fpca_spe"] = p.spe

        body = (x > 0.10 * L0) & (L0 - x > 60.0)
        v = float(np.polyfit(t[body], x[body], 1)[0]) if body.sum() > 100 else float("nan")
        xb = self.grid.x_of(L0)
        fit_mask = xb > 0.08 * L0
        th = fit_theta(xb[fit_mask], reg[fit_mask], L0, R, v, ci.billet_temp_C, self.alloy(ci.alloy_id), press)
        out.update({"theta_sigma_scale": th.sigma_scale, "theta_mu": th.mu, "theta_F_tool_N": th.F_tool_N,
                    "theta_resid_rms_N": th.resid_rms_N, "ram_speed_mm_s": v})
        h_up, lam = upturn_location(x, f, L0, self.h_stop_mm, press.amplitude_at_onset_N)
        out["upturn_h_mm"], out["upturn_lam_mm"] = h_up, lam
        out["upturn_detected"] = bool(np.isfinite(h_up))
        out["peak_force_N"] = float(np.max(f))

        liner = np.array(ci.liner_temps_C, dtype=float)
        out.update({
            "billet_temp_C": ci.billet_temp_C,
            "billet_length_mm": ci.billet_length_mm,
            "liner_temp_mean_C": float(liner.mean()),
            "liner_temp_spread_K": float(liner.max() - liner.min()),
            "dT_K": ci.billet_temp_C - float(liner.mean()),
            # cross-project features
            "oil_temp_C": ci.oil_temp_C,
            "supply_pressure_min_bar": ci.supply_pressure_min_bar,
            "pump_energy_kwh": ci.pump_energy_kwh,
            "shear_stroke_s": ci.phase_durations_s.get("shear_stroke", float("nan")),
            "dead_cycle_s": float(sum(ci.phase_durations_s.values())) if ci.phase_durations_s else float("nan"),
            **{f"phase_{k}_s": float(val) for k, val in ci.phase_durations_s.items()},
        })
        return out
