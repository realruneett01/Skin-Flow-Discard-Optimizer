"""Forward force-curve model of a direct aluminium extrusion stroke (Task 1.1, layers L1-L2).

Units used throughout: mm, s, N, MPa, K. Pressures are in bar only at the sensor
interface (1 MPa = 10 bar, and MPa * mm^2 = N).

Plan section 1.2 equations::

    eps_dot = 6 * v_ram * Db^2 * ln(R) / (Db^3 - De^3)          (Feltham mean strain rate)
    Z       = eps_dot * exp(Q / (Rg * T))                        (Zener-Hollomon)
    sigma_f = (1/alpha) * asinh((Z/A)^(1/n))                     (Sellars-Tegart)
    F(x)    = Ac * sigma_f * [ln(R) + 4*mu*(L0 - x)/Db] + F_tool + F_up(h)
    h       = L0 - x,     F_up(h) = a_up * exp(-h / lam)

Verification notes (plan rule 7, also in docs/assumptions.md):

* ``Db`` is taken as the container bore, because after upsetting the billet fills
  the container. ``L0`` is the upset billet length (volume conserved).
* ``De`` is the equivalent round diameter ``Db / sqrt(R)``.
* The formula has no redundant-work term (Johnson's ``a + b ln R``), so it
  underestimates absolute force for real dies. ``sigma_scale`` (theta[0]) absorbs
  that. The sign conventions check out: friction falls as the ram advances, and
  ``F_up`` grows as the remaining thickness ``h`` goes to zero.
* The simulator adds two terms the plan formula leaves out, both confined to the
  start of the stroke: a container-fill ramp and a dummy-block entry
  (breakthrough) bump. They let Task 2.1's gate be tested against something real.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import numpy as np

from skinflow_discard_optimizer.config import iter_leaves, load_config

RG = 8.314462618  # J/(mol K), CODATA 2018 molar gas constant
KELVIN = 273.15
BAR_PER_MPA = 10.0

Shape = Literal["upturn", "drop"]


# --------------------------------------------------------------------------- parameters

@dataclass(frozen=True)
class Alloy:
    name: str
    Q: float           # J/mol
    alpha: float       # 1/MPa
    n: float           # -
    lnA: float         # ln(1/s)
    density: float     # kg/m^3
    mu_nominal: float  # -

    @classmethod
    def from_config(cls, name: str) -> "Alloy":
        tree = load_config("alloys")
        if name not in tree:
            raise KeyError(f"alloy {name!r} not in config/alloys.yaml")
        leaves = {k: float(leaf["value"]) for k, leaf in iter_leaves(tree[name])}
        return cls(name=name, **leaves)


def _flat_values(tree: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for dotted, leaf in iter_leaves(tree):
        key = dotted.rsplit(".", 1)[-1]
        if key in out:
            raise ValueError(f"duplicate parameter name {key!r} in config")
        out[key] = float(leaf["value"])
    return out


@dataclass(frozen=True)
class Press:
    """Press geometry, hydraulics, nominal process values and sensor noise (config/press.yaml)."""

    container_bore_mm: float
    billet_diameter_mm: float
    billet_length_mm: float
    billet_length_sd_mm: float
    extrusion_ratio: float
    cap_area_mm2: float
    rod_area_mm2: float
    rod_back_pressure_bar: float
    rated_force_N: float
    pump_overall_efficiency: float
    supply_pressure_nominal_bar: float
    ram_speed_mm_s: float
    ram_speed_sd_mm_s: float
    accel_time_constant_s: float
    stroke_end_thickness_mm: float
    billet_temp_C: float
    billet_temp_sd_C: float
    taper_K: float
    deformation_heating_K: float
    heating_length_mm: float
    liner_temp_C: float
    F_tool_N: float
    fill_length_mm: float
    entry_amplitude_frac: float
    entry_length_mm: float
    amplitude_at_onset_N: float
    decay_length_mm: float
    drop_fraction: float
    sample_rate_hz: float
    pressure_noise_bar: float
    position_noise_mm: float
    temp_noise_C: float

    @classmethod
    def load(cls) -> "Press":
        return cls(**_flat_values(load_config("press")))

    @property
    def container_area_mm2(self) -> float:
        return np.pi / 4.0 * self.container_bore_mm**2

    def upset_length(self, billet_length_mm: float) -> float:
        """Billet length after it is upset to fill the container (volume conserved)."""
        return billet_length_mm * (self.billet_diameter_mm / self.container_bore_mm) ** 2


@dataclass(frozen=True)
class StrokeSpec:
    """Everything that defines one extrusion stroke. In the simulator these are ground truth."""

    alloy: Alloy
    L0_mm: float                  # upset billet length
    extrusion_ratio: float
    ram_speed_mm_s: float
    T_front_C: float              # billet temperature at the die end
    taper_K: float
    heating_K: float
    heating_length_mm: float
    sigma_scale: float            # theta[0]
    mu: float                     # theta[1]
    F_tool_N: float               # theta[2]
    h_onset_mm: float             # thickness where the end-of-stroke change starts
    upturn_amp_N: float           # size of F_up at h_onset ("upturn" shape)
    lam_mm: float
    shape: Shape = "upturn"
    drop_fraction: float = 0.08   # size of the fall as a fraction of force at onset ("drop" shape)
    fill_length_mm: float = 3.0
    entry_amplitude_frac: float = 0.06
    entry_length_mm: float = 8.0
    accel_tau_s: float = 0.5
    h_end_mm: float = 8.0

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.sigma_scale, self.mu, self.F_tool_N])

    @property
    def stroke_length_mm(self) -> float:
        return self.L0_mm - self.h_end_mm


def nominal_stroke(alloy: str | Alloy = "AA6063", press: Press | None = None, **overrides) -> StrokeSpec:
    """A stroke at the nominal values in config/press.yaml, with optional field overrides."""
    press = press or Press.load()
    alloy = alloy if isinstance(alloy, Alloy) else Alloy.from_config(alloy)
    spec = StrokeSpec(
        alloy=alloy,
        L0_mm=press.upset_length(press.billet_length_mm),
        extrusion_ratio=press.extrusion_ratio,
        ram_speed_mm_s=press.ram_speed_mm_s,
        T_front_C=press.billet_temp_C,
        taper_K=press.taper_K,
        heating_K=press.deformation_heating_K,
        heating_length_mm=press.heating_length_mm,
        sigma_scale=1.0,
        mu=alloy.mu_nominal,
        F_tool_N=press.F_tool_N,
        h_onset_mm=30.0,
        upturn_amp_N=press.amplitude_at_onset_N,
        lam_mm=press.decay_length_mm,
        drop_fraction=press.drop_fraction,
        fill_length_mm=press.fill_length_mm,
        entry_amplitude_frac=press.entry_amplitude_frac,
        entry_length_mm=press.entry_length_mm,
        accel_tau_s=press.accel_time_constant_s,
        h_end_mm=press.stroke_end_thickness_mm,
    )
    return replace(spec, **overrides)


# --------------------------------------------------------------------------- L1: flow stress

def feltham_strain_rate(v_ram_mm_s, Db_mm: float, R: float):
    """Mean strain rate (1/s). Lengths in mm, speed in mm/s."""
    De = Db_mm / np.sqrt(R)
    return 6.0 * np.asarray(v_ram_mm_s) * Db_mm**2 * np.log(R) / (Db_mm**3 - De**3)


def zener_hollomon(eps_dot, T_K, Q: float):
    """Temperature-compensated strain rate Z (1/s)."""
    return np.asarray(eps_dot) * np.exp(Q / (RG * np.asarray(T_K)))


def flow_stress_MPa(eps_dot, T_K, alloy: Alloy):
    """Sellars-Tegart flow stress in MPa, evaluated in log space to avoid overflow."""
    with np.errstate(divide="ignore"):
        ln_z = np.log(np.asarray(eps_dot, dtype=float)) + alloy.Q / (RG * np.asarray(T_K))
    return np.arcsinh(np.exp((ln_z - alloy.lnA) / alloy.n)) / alloy.alpha


def billet_temperature_K(x_mm, spec: StrokeSpec):
    """Bulk temperature of the deforming zone versus ram stroke: taper minus plus deformation heating."""
    x = np.asarray(x_mm, dtype=float)
    return (
        spec.T_front_C
        - spec.taper_K * x / spec.L0_mm
        + spec.heating_K * (1.0 - np.exp(-x / spec.heating_length_mm))
        + KELVIN
    )


# --------------------------------------------------------------------------- L2: force curve

def _parse_named_args(args: tuple, kwargs: dict, spec: list[tuple[str, any]]) -> dict:
    res = {}
    for i, (k, default) in enumerate(spec):
        if i < len(args):
            res[k] = args[i]
        else:
            res[k] = kwargs.get(k, default)
    return res


def base_force_N(*args, **kwargs):
    """Reduction + container-friction force (plan formula without F_tool and F_up)."""
    spec_list = [
        ("x_mm", None),
        ("sigma_MPa", None),
        ("spec_or_L0", None),
        ("Db_mm", None),
        ("R", None),
        ("mu", None),
        ("sigma_scale", 1.0),
    ]
    p = _parse_named_args(args, kwargs, spec_list)
    x_mm, sigma_MPa, spec_or_L0 = p["x_mm"], p["sigma_MPa"], p["spec_or_L0"]
    Db_mm, R, mu, sigma_scale = p["Db_mm"], p["R"], p["mu"], p["sigma_scale"]
    L0 = spec_or_L0.L0_mm if isinstance(spec_or_L0, StrokeSpec) else float(spec_or_L0)
    Ac = np.pi / 4.0 * Db_mm**2
    return Ac * sigma_scale * sigma_MPa * (np.log(R) + 4.0 * mu * (L0 - np.asarray(x_mm)) / Db_mm)


def upturn_coefficient(amp_at_onset_N: float, h_onset_mm: float, lam_mm: float) -> float:
    """``a_up`` such that ``a_up * exp(-h_onset/lam) == amp_at_onset``."""
    return amp_at_onset_N * np.exp(h_onset_mm / lam_mm)


def end_of_stroke_force_N(h_mm, spec: StrokeSpec, base_at_onset_N: float | None = None):
    """``F_up(h)``. Positive upturn (default), or the source spec's force drop when shape='drop'."""
    h = np.asarray(h_mm, dtype=float)
    if spec.shape == "upturn":
        a_up = upturn_coefficient(spec.upturn_amp_N, spec.h_onset_mm, spec.lam_mm)
        return a_up * np.exp(-h / spec.lam_mm)
    if spec.shape == "drop":
        if base_at_onset_N is None:
            raise ValueError("drop shape needs the baseline force at onset")
        drop = spec.drop_fraction * base_at_onset_N
        # Force falls smoothly once h passes below the onset, saturating at `drop`.
        return -drop * (1.0 - np.exp(-np.clip(spec.h_onset_mm - h, 0.0, None) / spec.lam_mm))
    raise ValueError(f"unknown shape {spec.shape!r}")


@dataclass
class ForceBreakdown:
    x_mm: np.ndarray
    h_mm: np.ndarray
    T_K: np.ndarray
    eps_dot: np.ndarray
    sigma_MPa: np.ndarray
    base_N: np.ndarray
    tool_N: np.ndarray
    entry_N: np.ndarray
    fill: np.ndarray
    end_N: np.ndarray
    total_N: np.ndarray = field(init=False)

    def __post_init__(self):
        self.total_N = self.fill * (self.base_N + self.tool_N + self.entry_N) + self.end_N


def _eval_steady_base_force(x_pos: float, spec: StrokeSpec, Db: float, R: float) -> float:
    eps = feltham_strain_rate(spec.ram_speed_mm_s, Db, R)
    sig = flow_stress_MPa(eps, billet_temperature_K(x_pos, spec), spec.alloy)
    return float(base_force_N(x_pos, sig, spec, Db, R, spec.mu, spec.sigma_scale))


def force_breakdown(x_mm, spec: StrokeSpec, press: Press, v_ram_mm_s=None) -> ForceBreakdown:
    """Evaluate every force term along the stroke. ``v_ram_mm_s`` defaults to the steady speed."""
    x = np.asarray(x_mm, dtype=float)
    v = spec.ram_speed_mm_s if v_ram_mm_s is None else np.asarray(v_ram_mm_s, dtype=float)
    Db, R = press.container_bore_mm, spec.extrusion_ratio
    h = spec.L0_mm - x
    T = billet_temperature_K(x, spec)
    eps = np.broadcast_to(feltham_strain_rate(v, Db, R), x.shape)
    sigma = flow_stress_MPa(eps, T, spec.alloy)
    base = base_force_N(x, sigma, spec, Db, R, spec.mu, spec.sigma_scale)
    tool = np.full_like(x, spec.F_tool_N)

    # Breakthrough bump scaled on the steady-state force at the start of the stroke.
    f0 = _eval_steady_base_force(0.0, spec, Db, R)
    xe = spec.entry_length_mm
    entry = spec.entry_amplitude_frac * f0 * (x / xe) * np.exp(1.0 - x / xe)
    fill = 1.0 - np.exp(-np.clip(x, 0.0, None) / spec.fill_length_mm)

    base_on = None
    if spec.shape == "drop":
        x_on = spec.L0_mm - spec.h_onset_mm
        base_on = _eval_steady_base_force(x_on, spec, Db, R)
    end = end_of_stroke_force_N(h, spec, base_on)
    return ForceBreakdown(x, h, T, np.asarray(eps), sigma, base, tool, entry, fill, end)


def ram_force_N(x_mm, spec: StrokeSpec, press: Press, v_ram_mm_s=None) -> np.ndarray:
    return force_breakdown(x_mm, spec, press, v_ram_mm_s).total_N


# --------------------------------------------------------------------------- hydraulics

def force_to_pressures(force_N, press: Press, p_rod_bar=None):
    """Cap and rod pressure (bar) that produce ``force_N``: F = p_cap*A_cap - p_rod*A_rod."""
    F = np.asarray(force_N, dtype=float)
    p_rod = np.full_like(F, press.rod_back_pressure_bar) if p_rod_bar is None else np.asarray(p_rod_bar)
    p_cap_MPa = (F + p_rod / BAR_PER_MPA * press.rod_area_mm2) / press.cap_area_mm2
    return p_cap_MPa * BAR_PER_MPA, p_rod


def pressures_to_force(p_cap_bar, p_rod_bar, press: Press):
    """Ram force (N) from measured cap and rod pressures (bar)."""
    return (np.asarray(p_cap_bar) * press.cap_area_mm2 - np.asarray(p_rod_bar) * press.rod_area_mm2) / BAR_PER_MPA


# --------------------------------------------------------------------------- stroke simulation

@dataclass
class StrokeData:
    """One simulated stroke sampled at the sensor rate. ``*_true`` are noise-free."""

    t_s: np.ndarray
    x_true_mm: np.ndarray
    x_mm: np.ndarray            # measured position
    v_mm_s: np.ndarray
    p_cap_bar: np.ndarray       # measured
    p_rod_bar: np.ndarray       # measured
    force_true_N: np.ndarray
    T_K: np.ndarray
    spec: StrokeSpec

    @property
    def h_true_mm(self) -> np.ndarray:
        return self.spec.L0_mm - self.x_true_mm

    def force_measured_N(self, press: Press) -> np.ndarray:
        return pressures_to_force(self.p_cap_bar, self.p_rod_bar, press)


def stall_limit_N(press: Press) -> float:
    """Largest ram force the hydraulics can deliver: supply pressure on the cap, back pressure on the rod."""
    return (press.supply_pressure_nominal_bar * press.cap_area_mm2
            - press.rod_back_pressure_bar * press.rod_area_mm2) / BAR_PER_MPA


def ram_kinematics(spec: StrokeSpec, fs_hz: float):
    """Time, position and speed for a first-order acceleration to the steady ram speed."""
    v, tau, S = spec.ram_speed_mm_s, spec.accel_tau_s, spec.stroke_length_mm
    t_end = S / v + tau  # x(t) ~ v*(t - tau) once accelerated
    t = np.arange(0.0, t_end + 5.0 / fs_hz, 1.0 / fs_hz)
    decay = np.exp(-t / tau)
    x = v * (t - tau * (1.0 - decay))
    keep = x <= S
    t, x = t[keep], x[keep]
    return t, x, v * (1.0 - decay[keep])


def simulate_stroke(*args, **kwargs) -> StrokeData:
    """Simulate cap/rod pressure and position at the sensor rate for one stroke.

    ``extra_force_N(x)`` adds a physical force term (e.g. a flash spike). ``cap_gain``,
    ``cap_bias_bar`` and ``encoder_offset_mm`` distort only the *measurements*.

    The press cannot push harder than its supply pressure allows. If the end-of-stroke
    upturn demands more, the ram stalls and the stroke ends there (``stall_limit_N``).
    """
    spec_list = [
        ("spec", None),
        ("press", None),
        ("rng", None),
        ("fs_hz", None),
        ("noise", True),
        ("extra_force_N", None),
        ("cap_gain", 1.0),
        ("cap_bias_bar", 0.0),
        ("encoder_offset_mm", 0.0),
    ]
    p = _parse_named_args(args, kwargs, spec_list)
    spec, press = p["spec"], p["press"]
    rng, fs_hz, noise = p["rng"], p["fs_hz"], p["noise"]
    extra_force_N = p["extra_force_N"]
    cap_gain, cap_bias_bar, encoder_offset_mm = p["cap_gain"], p["cap_bias_bar"], p["encoder_offset_mm"]

    fs = fs_hz or press.sample_rate_hz
    rng = rng if rng is not None else np.random.default_rng()
    t, x, v = ram_kinematics(spec, fs)
    br = force_breakdown(x, spec, press, v_ram_mm_s=np.maximum(v, 1e-6))
    force = br.total_N if extra_force_N is None else br.total_N + extra_force_N(x)
    over = np.flatnonzero((force > stall_limit_N(press)) & (x > 0.5 * spec.L0_mm))
    if over.size:
        n = over[0]
        t, x, v, force = t[:n], x[:n], v[:n], force[:n]
        br.T_K = br.T_K[:n]
    p_cap, p_rod = force_to_pressures(force, press)
    p_cap = p_cap * cap_gain + cap_bias_bar
    x_meas = x + encoder_offset_mm
    if noise:
        s = press.pressure_noise_bar
        p_cap = p_cap + rng.normal(0.0, s, p_cap.shape)
        p_rod = p_rod + rng.normal(0.0, s, p_rod.shape)
        x_meas = x_meas + rng.normal(0.0, press.position_noise_mm, x.shape)
    return StrokeData(t, x, x_meas, v, p_cap, p_rod, force, br.T_K, spec)
