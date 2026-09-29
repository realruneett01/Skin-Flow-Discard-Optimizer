"""Preprocessing of one stroke: filtering, derivatives, position resampling, onset gate (Task 2.1).

Pipeline (``preprocess_stroke``):

1. Force from cap and rod pressure; zero-phase low-pass on force and position.
2. Resample from time to the ram-position domain by bin-averaging onto a uniform
   grid, so the curve is indexed by stroke (mm), not time.
3. Savitzky-Golay first and second derivatives with the window picked from the
   noise level, per position (``adaptive_sg_derivative``, Lepski's method).
   ``choose_sg_window`` gives the single global MSE-optimal window for reports.
4. Detect the dummy-block entry transient and any flash spike inside the first 5%
   of the stroke from the data, not a fixed time (``detect_start_transients``,
   a robust fit of the generic start shape plus a residual test).
5. Gate: onset checks are allowed only where ``x >= 0.85 * L0`` *and* the start
   transient has decayed.

Zero-phase filtering uses future samples, so it serves post-stroke features and
plots. Anything that decides during the stroke uses the causal variants
(``sg_derivative(..., causal=True)``) or the UKF.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, savgol_coeffs, sosfiltfilt

from skinflow_discard_optimizer.sim.force_model import Press, pressures_to_force

GATE_FRACTION = 0.85
START_WINDOW_FRACTION = 0.05


# --------------------------------------------------------------------------- filtering

def lowpass(y: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (forward-backward, so no phase lag)."""
    sos = butter(order, cutoff_hz, fs=fs_hz, output="sos")
    return sosfiltfilt(sos, y)


def robust_noise_sd(y: np.ndarray) -> float:
    """White-noise standard deviation from the MAD of second differences.

    Second differences cancel any locally linear trend, so the estimate is barely
    affected by the signal itself. For white noise, var(d2) = 6 * sigma^2.
    """
    d2 = np.diff(np.asarray(y, dtype=float), n=2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


# --------------------------------------------------------------------------- resampling

@dataclass
class PositionCurve:
    x_mm: np.ndarray        # uniform grid (bin centres)
    force_N: np.ndarray     # bin-averaged force
    counts: np.ndarray      # samples per bin (0 where the ram did not pass)
    dx_mm: float


def resample_to_position(x_mm: np.ndarray, force_N: np.ndarray, dx_mm: float = 0.1,
                         **kwargs) -> PositionCurve:
    """Bin-average force onto a uniform ram-position grid. Empty bins are linearly filled."""
    x_min: float | None = kwargs.get("x_min", None)
    x_max: float | None = kwargs.get("x_max", None)
    x_min = float(np.min(x_mm)) if x_min is None else x_min
    x_max = float(np.max(x_mm)) if x_max is None else x_max
    edges = np.arange(x_min, x_max + dx_mm, dx_mm)
    idx = np.clip(np.digitize(x_mm, edges) - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    counts = np.bincount(idx, minlength=n).astype(float)
    sums = np.bincount(idx, weights=force_N, minlength=n)
    centres = 0.5 * (edges[:-1] + edges[1:])
    f = np.full(n, np.nan)
    ok = counts > 0
    f[ok] = sums[ok] / counts[ok]
    if (~ok).any() and ok.sum() >= 2:
        f[~ok] = np.interp(centres[~ok], centres[ok], f[ok])
    return PositionCurve(centres, f, counts, dx_mm)


# --------------------------------------------------------------------------- derivatives

def sg_derivative(y: np.ndarray, window: int, deriv: int, dx: float,
                  *args, **kwargs) -> np.ndarray:
    """Savitzky-Golay derivative. ``causal=True`` evaluates each point from past samples only."""
    polyorder: int = args[0] if len(args) > 0 else kwargs.get("polyorder", 3)
    causal: bool = args[1] if len(args) > 1 else kwargs.get("causal", False)
    if window % 2 == 0:
        window += 1
    pos = window - 1 if causal else None
    c = savgol_coeffs(window, polyorder, deriv=deriv, delta=dx, pos=pos, use="dot")
    y = np.asarray(y, dtype=float)
    if causal:
        out = np.full_like(y, np.nan)
        conv = np.convolve(y, c[::-1], mode="valid")
        out[window - 1:] = conv
        return out
    # symmetric: correlate, edges handled by padding with the edge value
    half = window // 2
    ypad = np.pad(y, half, mode="edge")
    return np.convolve(ypad, c[::-1], mode="valid")


def _coeff_norm(window: int, polyorder: int, deriv: int, dx: float) -> float:
    return float(np.linalg.norm(savgol_coeffs(window, polyorder, deriv=deriv, delta=dx, use="dot")))


@dataclass
class WindowChoice:
    window: int
    noise_sd: float
    candidates: np.ndarray
    mse: np.ndarray
    noise_var: np.ndarray
    bias2: np.ndarray


def choose_sg_window(y: np.ndarray, dx: float, deriv: int, polyorder: int = 3,
                     **kwargs) -> WindowChoice:
    """Pick the SG window minimising estimated MSE = noise variance + bias^2.

    * noise variance of the derivative is analytic: ``(sigma * ||c_W||)^2``, with
      sigma from ``robust_noise_sd``.
    * bias^2 is estimated against the smallest window W0 (nearly unbiased, noisy):
      E[(d_W - d_W0)^2] ~ bias_W^2 + sigma^2 * ||c_W - c_W0||^2, so the excess of the
      observed mean square difference over its noise part estimates bias_W^2.

    ``region`` restricts the MSE average to part of the curve (e.g. the tail).
    """
    noise_sd: float | None = kwargs.get("noise_sd", None)
    candidates = kwargs.get("candidates", None)
    region: np.ndarray | None = kwargs.get("region", None)

    y = np.asarray(y, dtype=float)
    sigma = robust_noise_sd(y) if noise_sd is None else noise_sd
    if candidates is None:
        lo = polyorder + 2 + (polyorder % 2 == 0)
        candidates = np.unique(np.clip(np.round(np.geomspace(lo, 401, 24)).astype(int) | 1, lo | 1, None))
    candidates = np.asarray(candidates)
    mask = np.ones(y.shape, bool) if region is None else np.asarray(region, bool)
    w0 = int(candidates[0])
    c0 = savgol_coeffs(w0, polyorder, deriv=deriv, delta=dx, use="dot")
    d0 = sg_derivative(y, w0, deriv, dx, polyorder)
    nv, b2 = [], []
    for w in candidates:
        cw = savgol_coeffs(int(w), polyorder, deriv=deriv, delta=dx, use="dot")
        pad = (len(cw) - len(c0)) // 2
        diff = cw.copy()
        diff[pad:pad + len(c0)] -= c0
        dw = sg_derivative(y, int(w), deriv, dx, polyorder)
        half = int(w) // 2
        m = mask.copy()
        m[:half] = False
        m[len(m) - half:] = False  # ignore edge-padded samples
        ms = float(np.mean((dw[m] - d0[m]) ** 2)) if m.any() else np.inf
        b2.append(max(ms - sigma**2 * float(diff @ diff), 0.0))
        nv.append(sigma**2 * float(cw @ cw))
    nv, b2 = np.array(nv), np.array(b2)
    mse = nv + b2
    return WindowChoice(int(candidates[np.argmin(mse)]), sigma, candidates, mse, nv, b2)


def adaptive_sg_derivative(y: np.ndarray, dx: float, deriv: int, noise_sd: float | None = None,
                           **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay derivative with a per-point window chosen by Lepski's method.

    Windows are tried from small to large. At each point the chosen window is the
    largest ``W`` whose estimate agrees with every smaller window ``w`` within
    ``kappa * (s_w + s_W)``, where ``s_w = sigma * ||c_w||`` is the analytic noise
    standard deviation of the window-``w`` derivative. Flat stretches get long windows
    (low noise) and sharp bends get short ones (low bias). The only input is the
    noise level. Returns ``(derivative, window_per_point)``.

    Defaults (tuned on simulated strokes, see tests/test_preprocess.py): windows from
    21 to 401 bins and ``kappa = 3.5``. Tiny windows add only noise at this noise
    level, and each extra candidate adds a chance of a spurious rejection.
    """
    polyorder: int = kwargs.get("polyorder", 3)
    candidates = kwargs.get("candidates", None)
    kappa: float = kwargs.get("kappa", 3.5)

    y = np.asarray(y, dtype=float)
    sigma = robust_noise_sd(y) if noise_sd is None else noise_sd
    if candidates is None:
        candidates = np.unique(np.round(np.geomspace(21, 401, 14)).astype(int) | 1)
    cands = [int(w) for w in candidates if w <= len(y)]
    ests = np.stack([sg_derivative(y, w, deriv, dx, polyorder) for w in cands])
    sds = np.array([sigma * _coeff_norm(w, polyorder, deriv, dx) for w in cands])
    chosen = np.zeros(len(y), dtype=int)
    alive = np.ones(len(y), bool)
    for j in range(1, len(cands)):
        ok = np.ones(len(y), bool)
        for i in range(j):
            ok &= np.abs(ests[j] - ests[i]) <= kappa * (sds[i] + sds[j])
        alive &= ok
        chosen[alive] = j
    out = ests[chosen, np.arange(len(y))]
    return out, np.asarray(cands)[chosen]


# --------------------------------------------------------------------------- start transients and gate

@dataclass
class StartTransients:
    flash_windows_mm: list[tuple[float, float]]
    transient_end_mm: float
    threshold_N: float
    peak_x_mm: float         # breakthrough peak position
    fit: np.ndarray          # fitted start-shape parameters [a, b, c, x_fill, x_entry, k]


def start_shape(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Expected start of a stroke: fill/acceleration ramp x (friction line + entry bump).

    The ramp ``1 - exp(-(x/x_fill)^k)`` covers both container fill and the ram
    accelerating (lower strain rate, lower flow stress) in the first few mm.
    """
    a, b, c, xf, xe, k = p
    return (1.0 - np.exp(-(x / xf) ** k)) * (a + b * x + c * (x / xe) * np.exp(1.0 - x / xe))


def _extract_flash_windows(resid: np.ndarray, xs: np.ndarray, thr: float, dx: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    above = resid > thr
    if above.any():
        idx = np.flatnonzero(above)
        splits = np.flatnonzero(np.diff(idx) > int(2.0 / dx)) + 1
        for grp in np.split(idx, splits):
            if len(grp) * dx >= 0.3:
                windows.append((float(xs[grp[0]] - 1.0), float(xs[grp[-1]] + 1.0)))
    return windows


def _compute_transient_end(fit_x: np.ndarray, xs: np.ndarray, sigma: float,
                           windows: list[tuple[float, float]]) -> float:
    a, b, c, xf, xe, k = fit_x
    bump = c * (xs / xe) * np.exp(1.0 - xs / xe)
    fill_gap = (a + b * xs) * np.exp(-(xs / xf) ** k)
    active = (bump > 3 * sigma) | (fill_gap > 3 * sigma)
    t_end = float(xs[np.flatnonzero(active)[-1]]) if active.any() else float(xs[0])
    if windows:
        t_end = max(t_end, max(w[1] for w in windows))
    return min(t_end, float(xs[-1])) if not windows else t_end


def detect_start_transients(curve: PositionCurve, L0_mm: float, k_sigma: float = 6.0,
                            noise_sd: float | None = None) -> StartTransients:
    """Find flash spikes and the end of the entry transient inside the first 5% of stroke."""
    from scipy.optimize import least_squares

    x, f, dx = curve.x_mm, curve.force_N, curve.dx_mm
    start_end = START_WINDOW_FRACTION * L0_mm
    ref = (x > 0.10 * L0_mm) & (x < 0.30 * L0_mm)
    sigma = robust_noise_sd(f[ref]) if noise_sd is None else noise_sd
    slope = float(np.polyfit(x[ref], f[ref], 1)[0]) if ref.sum() > 10 else 0.0

    m = x <= start_end
    xs, fs = x[m], f[m]
    a0 = float(np.median(fs[xs > 0.6 * start_end])) - slope * 0.8 * start_end
    c0 = max(float(np.max(fs)) - a0, 0.0)
    p0 = np.array([a0, slope, c0, 3.0, 8.0, 1.0])
    lb = [0.0, -np.inf, 0.0, 0.2, 2.0, 0.3]
    ub = [np.inf, np.inf, np.inf, 25.0, 40.0, 4.0]
    res_fn = lambda p: (start_shape(p, xs) - fs) / sigma  # noqa: E731
    fit = least_squares(res_fn, np.clip(p0, lb, ub), bounds=(lb, ub), x_scale="jac")
    fit = least_squares(res_fn, fit.x, bounds=(lb, ub), loss="soft_l1", f_scale=3.0, x_scale="jac")
    resid = fs - start_shape(fit.x, xs)
    thr = k_sigma * sigma

    windows = _extract_flash_windows(resid, xs, thr, dx)
    t_end = _compute_transient_end(fit.x, xs, sigma, windows)
    return StartTransients(windows, t_end, thr, float(fit.x[4]), fit.x)


def onset_gate(x_mm: np.ndarray, L0_mm: float, transient_end_mm: float,
               flash_windows_mm: list[tuple[float, float]] = ()) -> np.ndarray:
    """True where onset checks may run: late in the stroke, after the start transient, outside flashes."""
    x = np.asarray(x_mm)
    g = (x >= GATE_FRACTION * L0_mm) & (x > transient_end_mm)
    for a, b in flash_windows_mm:
        g &= ~((x >= a) & (x <= b))
    return g


# --------------------------------------------------------------------------- one-call pipeline

@dataclass
class PreprocessedStroke:
    curve: PositionCurve
    force_smooth_N: np.ndarray
    dF_dx: np.ndarray
    d2F_dx2: np.ndarray
    window_d1: np.ndarray      # SG window (bins) chosen at each position
    window_d2: np.ndarray
    transients: StartTransients
    gate: np.ndarray
    L0_mm: float
    meta: dict = field(default_factory=dict)

    @property
    def x_mm(self) -> np.ndarray:
        return self.curve.x_mm

    @property
    def h_mm(self) -> np.ndarray:
        return self.L0_mm - self.curve.x_mm


def preprocess_stroke(t_s: np.ndarray, x_mm: np.ndarray, p_cap_bar: np.ndarray, p_rod_bar: np.ndarray,
                      *args, **kwargs) -> PreprocessedStroke:
    """Full Task 2.1 pipeline for one stroke of raw sensor data."""
    press: Press = args[0] if len(args) > 0 else kwargs["press"]
    L0_mm: float = args[1] if len(args) > 1 else kwargs["L0_mm"]
    dx_mm: float = args[2] if len(args) > 2 else kwargs.get("dx_mm", 0.1)
    lp_cutoff_hz: float = args[3] if len(args) > 3 else kwargs.get("lp_cutoff_hz", 20.0)

    fs = 1.0 / float(np.median(np.diff(t_s)))
    force = pressures_to_force(p_cap_bar, p_rod_bar, press)
    x_f = lowpass(x_mm, fs, lp_cutoff_hz)
    curve = resample_to_position(x_f, force, dx_mm, x_min=0.0)
    sigma = robust_noise_sd(curve.force_N)
    smooth, w0 = adaptive_sg_derivative(curve.force_N, dx_mm, 0, sigma)
    d1, w1 = adaptive_sg_derivative(curve.force_N, dx_mm, 1, sigma)
    d2, w2 = adaptive_sg_derivative(curve.force_N, dx_mm, 2, sigma)
    tr = detect_start_transients(curve, L0_mm, noise_sd=sigma)
    gate = onset_gate(curve.x_mm, L0_mm, tr.transient_end_mm, tr.flash_windows_mm)
    return PreprocessedStroke(curve, smooth, d1, d2, w1, w2, tr, gate, L0_mm,
                              {"fs_hz": fs, "lp_cutoff_hz": lp_cutoff_hz, "noise_sd_N": sigma})
