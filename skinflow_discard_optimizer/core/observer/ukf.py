"""Within-stroke unscented Kalman filter for theta = [sigma_scale, mu, F_tool] (Task 3.1, layer L3).

Part of the Dead-Metal-Zone Observer.

Model (plan section 1.2)::

    y_k = F(x_k; theta_k) + v_k,     F = Ac*sigma(x)*s*[ln R + 4*mu*(L0 - x)/Db] + F_tool

``sigma(x)`` is the Sellars-Tegart flow stress at the measured billet temperature
(nominal taper and heating) and the ram speed estimated from past positions. The
measurement is ram force from cap and rod pressure. The baseline has no upturn
term, so innovations grow near the end of the stroke; the onset detector
(Task 3.2) tests exactly that.

Parameterisation (docs/assumptions.md A-23)
    The filter state is ``phi = [s, s*mu, F_tool]``, not theta. ``F`` is bilinear
    in theta (the ``s*mu`` product), and the data pin ``s*mu`` down far more tightly
    than ``s`` alone, so the posterior in theta is a curved ridge (a hyperbola) that
    a Gaussian in theta cannot represent. A UKF run directly on theta was tested and
    drifted well away from the exact posterior. In phi the measurement is linear, so
    the unscented update is exact. ``theta`` and its covariance are recovered from
    phi with an unscented transform. The random walk is on phi:
    ``phi_{k+1} = phi_k + w_k``.

Weak identifiability
    ``s`` and ``F_tool`` both move force almost uniformly along the stroke. Only
    the temperature-driven change of flow stress separates them, so they are
    individually uncertain even when the force prediction is tight. The posterior
    covariance reports this honestly. The cross-cycle prior (layer L6) is what
    narrows ``F_tool`` in practice.

Rates
    ``UKF.step`` handles one raw sample, so the filter can run live at 1 kHz.
    ``run_stroke(..., block=n)`` averages n raw samples per update (R scaled by
    1/n) for fast offline evaluation.

Consistency
    ``NIS = nu^2 / S`` is chi-square(1) when the model holds. The mean over windows
    of M updates is checked against chi-square(M)/M bounds (``nis_bounds``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from skinflow_discard_optimizer.paths import REPO_ROOT
from skinflow_discard_optimizer.sim.force_model import (
    KELVIN,
    Alloy,
    Press,
    feltham_strain_rate,
    flow_stress_MPa,
    pressures_to_force,
)

UKF_PARAMS_PATH = REPO_ROOT / "artifacts" / "ukf_params.json"
STATE_NAMES = ("sigma_scale", "mu", "F_tool_N")
PHI_SCALE = np.array([1.0, 1.0, 1e6])   # internal units: F_tool in MN
# Start of the filter window. Task 2.1 declares the entry transient over once it is
# below 3 sd of per-bin noise (~13 kN, around 5% of stroke), but the filter
# integrates hundreds of updates and sigma_scale/F_tool sit on a weak ridge, so even
# the few-kN tail of the breakthrough bump at 8-12% of stroke biased theta. From 12%
# the bias is gone on noise-free strokes (docs/assumptions.md A-24).
START_FRACTION = 0.12


@dataclass
class UKFParams:
    """Noise settings. Process noise is per mm of ram travel on phi = [s, s*mu, F_tool]."""

    q_per_mm: np.ndarray = field(default_factory=lambda: np.array([1e-7, 1e-7, 1e5]))
    force_noise_sd_N: float = 37_000.0       # per raw sample: pressure noise x effective area
    model_sd_N: float = 2_000.0              # unmodelled structure per update (does not average away)
    p0_sd: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.05, 0.3e6]))  # on theta

    def save(self, path: Path = UKF_PARAMS_PATH, **meta) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"q_per_mm": self.q_per_mm.tolist(), "force_noise_sd_N": self.force_noise_sd_N,
                                    "model_sd_N": self.model_sd_N, "p0_sd": self.p0_sd.tolist(), **meta}, indent=2))

    @classmethod
    def load(cls, path: Path = UKF_PARAMS_PATH) -> "UKFParams":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(np.array(d["q_per_mm"]), d["force_noise_sd_N"], d["model_sd_N"], np.array(d["p0_sd"]))

    @classmethod
    def for_press(cls, press: Press, path: Path = UKF_PARAMS_PATH) -> "UKFParams":
        p = cls.load(path)
        p.force_noise_sd_N = press.pressure_noise_bar / 10 * float(np.hypot(press.cap_area_mm2, press.rod_area_mm2))
        return p


@dataclass
class StrokeContext:
    """Per-billet quantities the measurement function needs (all measured or configured)."""

    L0_mm: float
    extrusion_ratio: float
    billet_temp_C: float
    alloy: Alloy
    press: Press

    def flow_stress(self, x_mm, v_mm_s: float):
        p = self.press
        x = np.asarray(x_mm)
        T = (self.billet_temp_C - p.taper_K * x / self.L0_mm
             + p.deformation_heating_K * (1 - np.exp(-x / p.heating_length_mm)) + KELVIN)
        return flow_stress_MPa(feltham_strain_rate(max(v_mm_s, 0.1), p.container_bore_mm, self.extrusion_ratio),
                               T, self.alloy)

    def regressors_many(self, x_mm: np.ndarray, v_mm_s: np.ndarray) -> np.ndarray:
        """Vectorised ``regressors`` for arrays of positions and speeds: shape (n, 3)."""
        p = self.press
        x = np.asarray(x_mm, float)
        T = (self.billet_temp_C - p.taper_K * x / self.L0_mm
             + p.deformation_heating_K * (1 - np.exp(-x / p.heating_length_mm)) + KELVIN)
        eps = feltham_strain_rate(np.maximum(np.asarray(v_mm_s, float), 0.1), p.container_bore_mm,
                                  self.extrusion_ratio)
        sig = flow_stress_MPa(eps, T, self.alloy)
        Ac, Db = p.container_area_mm2, p.container_bore_mm
        return np.column_stack([Ac * sig * np.log(self.extrusion_ratio), Ac * sig * 4.0 * (self.L0_mm - x) / Db,
                                np.ones_like(x)])

    def regressors(self, x_mm: float, v_mm_s: float) -> np.ndarray:
        """``H`` with ``F = H @ phi``: [Ac*sigma*lnR, Ac*sigma*4*(L0-x)/Db, 1]."""
        p = self.press
        sig = float(self.flow_stress(x_mm, v_mm_s))
        Ac, Db = p.container_area_mm2, p.container_bore_mm
        return np.array([Ac * sig * np.log(self.extrusion_ratio), Ac * sig * 4.0 * (self.L0_mm - x_mm) / Db, 1.0])


# --------------------------------------------------------------------------- unscented machinery

def _parse_named_args(args: tuple, kwargs: dict, spec: list[tuple[str, any]]) -> dict:
    res = {}
    for i, (k, default) in enumerate(spec):
        if i < len(args):
            res[k] = args[i]
        else:
            res[k] = kwargs.get(k, default)
    return res


def _weights(n: int, alpha: float, beta: float, kappa: float):
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    wm = np.full(2 * n + 1, 1.0 / (2 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1 - alpha**2 + beta)
    return c, wm, wc


def _sigma_points(m: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
    L = np.linalg.cholesky(c * (P + 1e-12 * np.eye(len(m))))
    return np.vstack([m, m + L.T, m - L.T])


def unscented_transform(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of ``fn(X)`` for X ~ N(m, P); ``fn`` maps (2n+1, n) -> (2n+1, d)."""
    spec = [("m", None), ("P", None), ("fn", None), ("alpha", 1.0), ("beta", 2.0), ("kappa", 0.0)]
    p = _parse_named_args(args, kwargs, spec)
    m, P, fn = p["m"], p["P"], p["fn"]
    c, wm, wc = _weights(len(m), p["alpha"], p["beta"], p["kappa"])
    Y = fn(_sigma_points(m, P, c))
    mu = wm @ Y
    d = Y - mu
    return mu, (wc * d.T) @ d


def phi_to_theta(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X[:, 0], X[:, 1] / X[:, 0], X[:, 2]])


def theta_to_phi(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X[:, 0], X[:, 0] * X[:, 1], X[:, 2]])


class UKF:
    """Unscented Kalman filter on scaled phi with a random-walk model and scalar measurements."""

    def __init__(self, *args, **kwargs):
        spec = [("theta0", None), ("P0_theta", None), ("params", None),
                ("alpha", 1.0), ("beta", 2.0), ("kappa", 0.0)]
        p = _parse_named_args(args, kwargs, spec)
        theta0, P0_theta, params = p["theta0"], p["P0_theta"], p["params"]
        m, P = unscented_transform(np.asarray(theta0, float), np.asarray(P0_theta, float), theta_to_phi)
        self.x = m / PHI_SCALE
        self.P = P / np.outer(PHI_SCALE, PHI_SCALE)
        self.q = params.q_per_mm / PHI_SCALE**2
        self.params = params
        self.c, self.wm, self.wc = _weights(3, p["alpha"], p["beta"], p["kappa"])

    @property
    def phi(self) -> np.ndarray:
        return self.x * PHI_SCALE

    @property
    def phi_cov(self) -> np.ndarray:
        return self.P * np.outer(PHI_SCALE, PHI_SCALE)

    def theta(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance of theta = [s, mu, F_tool] (unscented transform of phi)."""
        return unscented_transform(self.phi, self.phi_cov, phi_to_theta)

    def predict(self, dx_mm: float) -> None:
        # identity dynamics: sigma-point propagation reduces exactly to P += Q
        self.P = self.P + np.diag(self.q * max(dx_mm, 0.0))

    def predict_measurement(self, H: np.ndarray) -> float:
        return float(H @ self.phi)

    def update(self, y: float, H: np.ndarray, r_var: float) -> tuple[float, float]:
        """Unscented measurement update for y = H @ phi + v. Returns (innovation, S) in N, N^2."""
        X = _sigma_points(self.x, self.P, self.c)
        Y = X @ (H * PHI_SCALE)
        y_hat = self.wm @ Y
        dY = Y - y_hat
        S = float(self.wc @ (dY * dY)) + r_var
        C = (self.wc * dY) @ (X - self.x)
        K = C / S
        nu = float(y - y_hat)
        self.x = self.x + K * nu
        self.P = self.P - np.outer(K, K) * S
        self.P = 0.5 * (self.P + self.P.T)
        return nu, S

    def step(self, *args, **kwargs) -> tuple[float, float]:
        """One live update from a raw sample."""
        spec = [("y", None), ("x_mm", None), ("dx_mm", None), ("v_mm_s", None), ("ctx", None), ("r_var", None)]
        p = _parse_named_args(args, kwargs, spec)
        self.predict(p["dx_mm"])
        return self.update(p["y"], p["ctx"].regressors(p["x_mm"], p["v_mm_s"]), p["r_var"])


# --------------------------------------------------------------------------- one stroke

@dataclass
class UKFTrace:
    x_mm: np.ndarray
    L0_mm: float
    theta: np.ndarray          # (N, 3) posterior mean after each update
    theta_sd: np.ndarray       # (N, 3)
    innovation: np.ndarray     # (N,)
    S: np.ndarray              # (N,)
    y: np.ndarray              # measured force per update
    y_pred: np.ndarray         # one-step-ahead predicted force
    block: int
    final_phi: np.ndarray = field(default_factory=lambda: np.zeros(3))
    final_phi_cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    H: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # regressors per update
    r_var: float = 0.0                                                 # measurement variance per update

    @property
    def nis(self) -> np.ndarray:
        return self.innovation**2 / self.S

    @property
    def h_mm(self) -> np.ndarray:
        return self.L0_mm - self.x_mm


def nis_bounds(window: int, alpha: float = 0.01) -> tuple[float, float]:
    """Two-sided (1 - alpha) bounds on the mean of ``window`` chi-square(1) NIS values."""
    return chi2.ppf(alpha / 2, window) / window, chi2.ppf(1 - alpha / 2, window) / window


def windowed_nis(nis: np.ndarray, window: int) -> np.ndarray:
    """Mean NIS over consecutive non-overlapping windows."""
    n = len(nis) // window
    return nis[: n * window].reshape(n, window).mean(axis=1)


SPEED_WINDOW_S = 1.0


def causal_speed(t_s: np.ndarray, x_mm: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope of position over the previous ``window`` samples (past only).

    Speed feeds the flow stress, so noise in it becomes noise in the regressor
    (errors-in-variables). With ``s`` and ``F_tool`` this weakly separable, that
    biases theta badly. A two-point difference over 0.25 s is ~1% noisy; this
    least-squares slope over 1 s is ~50x quieter. Entries before ``window`` are NaN.
    """
    t = np.asarray(t_s, float) - t_s[0]
    x = np.asarray(x_mm, float)
    c = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    St, Sx, Stt, Stx = c(t), c(x), c(t * t), c(t * x)
    out = np.full(len(x), np.nan)
    i = np.arange(window - 1, len(x))
    a, b = i - window + 1, i + 1
    n = float(window)
    st, sx, stt, stx = St[b] - St[a], Sx[b] - Sx[a], Stt[b] - Stt[a], Stx[b] - Stx[a]
    out[i] = (n * stx - st * sx) / (n * stt - st * st)
    return out


def default_prior(alloy: Alloy, press: Press, params: UKFParams) -> tuple[np.ndarray, np.ndarray]:
    theta0 = np.array([1.0, alloy.mu_nominal, press.F_tool_N])
    return theta0, np.diag(params.p0_sd**2)


@dataclass
class StrokeBlocks:
    """Block-averaged filter inputs for one stroke (shared by the CPU filter and the GPU batch)."""

    xb: np.ndarray       # mean position per block
    yb: np.ndarray       # mean force per block
    v: np.ndarray        # causal speed at the end of each block
    H: np.ndarray        # regressor rows (n, 3)
    r_var: float         # measurement variance per update


def stroke_blocks(*args, **kwargs) -> StrokeBlocks:
    """Force, position, speed and regressors per filter update, from ``x_start`` to ``h_stop``."""
    spec = [
        ("t_s", None),
        ("x_mm", None),
        ("p_cap_bar", None),
        ("p_rod_bar", None),
        ("ctx", None),
        ("params", None),
        ("block", 1),
        ("x_start_mm", None),
        ("h_stop_mm", 0.0),
    ]
    p = _parse_named_args(args, kwargs, spec)
    t_s, x_mm = p["t_s"], p["x_mm"]
    p_cap_bar, p_rod_bar = p["p_cap_bar"], p["p_rod_bar"]
    ctx, params = p["ctx"], p["params"]
    block, x_start_mm, h_stop_mm = p["block"], p["x_start_mm"], p["h_stop_mm"]
    force = pressures_to_force(p_cap_bar, p_rod_bar, ctx.press)
    x_start = START_FRACTION * ctx.L0_mm if x_start_mm is None else x_start_mm
    fs = 1.0 / float(np.median(np.diff(t_s)))
    lag = max(int(SPEED_WINDOW_S * fs), 2)
    i0 = max(int(np.searchsorted(x_mm, x_start)), lag)
    stop = (ctx.L0_mm - x_mm) < h_stop_mm
    i1 = int(np.argmax(stop)) if stop.any() else len(x_mm)
    n = (i1 - i0) // block
    if n <= 0:
        raise ValueError("stroke too short for the requested window")
    sl = slice(i0, i0 + n * block)
    yb = force[sl].reshape(n, block).mean(axis=1)
    xb = x_mm[sl].reshape(n, block).mean(axis=1)
    last = np.arange(i0 + block - 1, i0 + n * block, block)          # last raw index in each block
    v = causal_speed(t_s, x_mm, lag)[last]
    r_var = params.force_noise_sd_N**2 / block + params.model_sd_N**2
    return StrokeBlocks(xb, yb, v, ctx.regressors_many(xb, v), r_var)


def run_stroke(*args, **kwargs) -> UKFTrace:
    """Run the filter causally over one stroke, from ``x_start`` until ``h`` reaches ``h_stop``.

    Ram speed at each update is the least-squares slope of position over the
    previous second (past samples only). ``theta_every`` computes the theta
    transform only every n-th update (it is the costly part); rows in between
    repeat the last value.

    ``freeze_h_mm``: once the remaining thickness falls below this, the filter stops
    updating and only predicts. The innovations are then residuals against the frozen
    baseline, with S = H P H' + R. The onset detector uses this so the baseline
    parameters cannot absorb the end-of-stroke upturn they are meant to reveal.
    """
    spec = [
        ("t_s", None),
        ("x_mm", None),
        ("p_cap_bar", None),
        ("p_rod_bar", None),
        ("ctx", None),
        ("params", None),
        ("block", 1),
        ("x_start_mm", None),
        ("h_stop_mm", 0.0),
        ("prior", None),
        ("theta_every", 1),
        ("freeze_h_mm", None),
    ]
    p = _parse_named_args(args, kwargs, spec)
    t_s, x_mm = p["t_s"], p["x_mm"]
    p_cap_bar, p_rod_bar = p["p_cap_bar"], p["p_rod_bar"]
    ctx = p["ctx"]
    params = p["params"] or UKFParams.for_press(ctx.press)
    block = p["block"]
    x_start_mm, h_stop_mm = p["x_start_mm"], p["h_stop_mm"]
    prior, theta_every, freeze_h_mm = p["prior"], p["theta_every"], p["freeze_h_mm"]

    sb = stroke_blocks(t_s, x_mm, p_cap_bar, p_rod_bar, ctx, params, block, x_start_mm, h_stop_mm)
    xb, yb, Hs, r_var = sb.xb, sb.yb, sb.H, sb.r_var
    n = len(xb)

    theta0, P0 = prior if prior is not None else default_prior(ctx.alloy, ctx.press, params)
    f = UKF(theta0, P0, params)
    th, sd = np.empty((n, 3)), np.empty((n, 3))
    nu, S, yp = np.empty(n), np.empty(n), np.empty(n)
    xprev = xb[0]
    m_th, sd_th = None, None
    for k in range(n):
        f.predict(xb[k] - xprev)
        xprev = xb[k]
        H = Hs[k]
        yp[k] = f.predict_measurement(H)
        if freeze_h_mm is not None and ctx.L0_mm - xb[k] < freeze_h_mm:
            nu[k] = yb[k] - yp[k]
            S[k] = float(H @ f.phi_cov @ H) + r_var
        else:
            nu[k], S[k] = f.update(yb[k], H, r_var)
        if k % theta_every == 0 or k == n - 1:
            m_th, cov = f.theta()
            sd_th = np.sqrt(np.clip(np.diag(cov), 0, None))
        th[k], sd[k] = m_th, sd_th
    return UKFTrace(xb, ctx.L0_mm, th, sd, nu, S, yb, yp, block, f.phi, f.phi_cov, Hs, r_var)


def tune_process_noise(strokes: list[tuple], params: UKFParams, *args, **kwargs) -> tuple[UKFParams, float]:
    """Maximum-likelihood process noise (per mm, on phi) and model noise from healthy strokes.

    ``strokes`` holds ``(t, x, p_cap, p_rod, ctx)`` tuples. The objective is the
    innovation negative log-likelihood, sum of 0.5*(log 2*pi*S + nu^2/S), over the
    stretch before the end-of-stroke region (``x < x_stop_frac * L0``), where the
    baseline model is meant to hold. Returns the tuned params and the minimum NLL.
    """
    from scipy.optimize import minimize
    block = args[0] if len(args) > 0 else kwargs.get("block", 50)
    x_stop_frac = args[1] if len(args) > 1 else kwargs.get("x_stop_frac", 0.85)
    maxiter = args[2] if len(args) > 2 else kwargs.get("maxiter", 300)

    def nll(logq: np.ndarray) -> float:
        p = UKFParams(np.exp(logq[:3]), params.force_noise_sd_N, float(np.exp(logq[3])), params.p0_sd)
        total = 0.0
        for t, x, pc, pr, ctx in strokes:
            tr = run_stroke(t, x, pc, pr, ctx, p, block=block, h_stop_mm=(1 - x_stop_frac) * ctx.L0_mm,
                            theta_every=10**9)
            total += 0.5 * float(np.sum(np.log(2 * np.pi * tr.S) + tr.innovation**2 / tr.S))
        return total

    x0 = np.log(np.r_[params.q_per_mm, params.model_sd_N])
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 0.05, "fatol": 0.5})
    return UKFParams(np.exp(res.x[:3]), params.force_noise_sd_N, float(np.exp(res.x[3])), params.p0_sd), float(res.fun)
