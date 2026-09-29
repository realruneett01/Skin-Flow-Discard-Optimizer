"""Skin-flow onset detection on UKF residuals (Task 3.2, layer L4).

Part of the Dead-Metal-Zone Observer.

After the gate (Task 2.1: ``x >= 0.85*L0`` and past the start transient) the UKF is
frozen and predicts the baseline force only. Its residuals ``r_k`` with variance
``S_k`` feed two detectors:

1. **Sequential GLR test** (primary).
   H0: ``r ~ N(0, S)``; H1: ``r = a*exp(-h/lam) + N(0, S)`` with unknown ``a >= 0``
   and ``lam``. For fixed ``lam`` the MLE of ``a`` is closed-form, so running sums
   ``B = sum g*r/S`` and ``C = sum g^2/S`` (``g = exp(-h/lam)``) per ``lam`` on a grid
   give the log-likelihood ratio ``B^2/(2C)`` in O(grid) per update. An alarm is
   raised when the maximum over the grid passes ``threshold``.

   The same sums give the **posterior over the onset position**. The onset is
   defined as the thickness where the upturn reaches ``A_ref`` (docs/assumptions.md
   A-19), so ``a = A_ref*exp(h_on/lam)`` and
   ``log L(h_on, lam) = a*B - a^2*C/2``, evaluated on an ``h_on x lam`` grid and
   marginalised over ``lam``. Because the rise is visible before ``F_up`` reaches
   ``A_ref``, the posterior sharpens *before* the ram gets to the onset.

2. **Bayesian online change-point detection** (Adams & MacKay 2007) on the
   standardised residuals ``z = r/sqrt(S)``, with a Normal model of unknown mean
   and known unit variance. It makes no assumption about the shape or the sign
   of the departure, so it also catches the source spec's "force drops" shape.
   It fires when the posterior probability of a change within the last
   ``recent`` updates exceeds ``p_fire``.

Confidence: "high" when both fire within ``agree_mm`` of ram travel of each other,
"low" when only one fires or they disagree, "none" when neither fires.

Baseline for comparison: ``naive_second_derivative`` thresholds a causal
Savitzky-Golay second derivative of force, with a constant offset calibrated on
training strokes to turn "first departure" into the defined onset position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from skinflow_discard_optimizer.core.preprocess import resample_to_position, sg_derivative

LAM_GRID = np.geomspace(2.5, 15.0, 64)   # fine: late in the stroke the data pin lam down tightly
H_GRID = np.arange(5.0, 90.0, 0.1)


def _parse_named_args(args: tuple, kwargs: dict, spec: list[tuple[str, any]]) -> dict:
    res = {}
    for i, (k, default) in enumerate(spec):
        if i < len(args):
            res[k] = args[i]
        else:
            res[k] = kwargs.get(k, default)
    return res


def observable_onset(h_onset_mm: float, lam_mm: float, amp_N: float, a_ref_N: float) -> float:
    """Thickness where ``F_up`` reaches ``a_ref`` for a simulated stroke.

    The simulator places ``h_onset`` where the upturn reaches that billet's own
    (randomised) amplitude. The force can only reveal where it reaches the fixed
    reference amplitude, which is this value. The gap between the two is real
    billet-to-billet noise between the signal and ``h_crit``.
    """
    return h_onset_mm + lam_mm * np.log(amp_N / a_ref_N)


class OnsetGLR:
    """Sequential GLR test and grid posterior over the onset position."""

    def __init__(self, *args, **kwargs):
        spec = [
            ("a_ref_N", None),
            ("threshold", 12.0),
            ("lam_grid", LAM_GRID),
            ("h_grid", H_GRID),
            ("h_prior", None),
            ("lam_log_prior", None),
            ("baseline_cov", None),
            ("noise_var", None),
        ]
        p = _parse_named_args(args, kwargs, spec)
        self.a_ref = p["a_ref_N"]
        self.threshold = p["threshold"]
        self.lam = np.asarray(p["lam_grid"], float)
        self.h = np.asarray(p["h_grid"], float)
        nl = len(self.lam)
        self.Bg = np.zeros(nl)
        self.Cgg = np.zeros(nl)
        self.marginal = p["baseline_cov"] is not None
        if self.marginal:
            self._init_marginal(p["baseline_cov"], p["noise_var"], nl)
        h_prior = p["h_prior"]
        self.log_prior_h = np.zeros_like(self.h) if h_prior is None else np.log(np.maximum(h_prior, 1e-300))
        lam_log_prior = p["lam_log_prior"]
        self.log_prior_lam = np.zeros_like(self.lam) if lam_log_prior is None else lam_log_prior
        self._a = self.a_ref * np.exp(np.clip(self.h[:, None] / self.lam[None, :], None, 60.0))
        self.stat = 0.0
        self.fired_at_h: float | None = None
        self.n = 0

    def _init_marginal(self, baseline_cov, noise_var, nl: int) -> None:
        if noise_var is None:
            raise ValueError("marginalised mode needs noise_var")
        self.R = float(noise_var)
        self._scale = np.array([1.0, 1.0, 1e6])       # work in scaled phi (F_tool in MN)
        P = np.asarray(baseline_cov) / np.outer(self._scale, self._scale)
        self.Pinv = np.linalg.inv(P + 1e-15 * np.eye(3))
        self.cgH = np.zeros((nl, 3))
        self.MHH = np.zeros((3, 3))
        self.bH = np.zeros(3)

    def update(self, *args, **kwargs) -> float:
        spec = [("h_mm", None), ("r", None), ("S", None), ("H", None)]
        p = _parse_named_args(args, kwargs, spec)
        h_mm, r, S, H = p["h_mm"], p["r"], p["S"], p["H"]
        g = np.exp(-h_mm / self.lam)
        if self.marginal:
            if H is None:
                raise ValueError("marginalised mode needs the regressor row H")
            Hs = np.asarray(H) * self._scale
            self.Bg += g * r / self.R
            self.Cgg += g * g / self.R
            self.cgH += np.outer(g, Hs) / self.R
            self.MHH += np.outer(Hs, Hs) / self.R
            self.bH += Hs * r / self.R
        else:
            self.Bg += g * r / S
            self.Cgg += g * g / S
        self.n += 1
        B, C = self._effective()
        llr = np.where(B > 0, B**2 / (2 * np.maximum(C, 1e-300)), 0.0)
        self.stat = float(llr.max())
        if self.fired_at_h is None and self.stat > self.threshold:
            self.fired_at_h = h_mm
        return self.stat

    def _effective(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.marginal:
            return self.Bg, self.Cgg
        Ainv_b = np.linalg.solve(self.Pinv + self.MHH, self.bH)
        Ainv_c = np.linalg.solve(self.Pinv + self.MHH, self.cgH.T)          # (3, n_lam)
        B = self.Bg - self.cgH @ Ainv_b
        C = self.Cgg - np.einsum("lj,jl->l", self.cgH, Ainv_c)
        return B, C

    def log_posterior(self) -> np.ndarray:
        """Unnormalised log posterior over (h_on, lam)."""
        B, C = self._effective()
        ll = self._a * B[None, :] - 0.5 * self._a**2 * C[None, :]
        return ll + self.log_prior_h[:, None] + self.log_prior_lam[None, :]

    def posterior(self) -> np.ndarray:
        """Posterior over ``h_grid`` (onset thickness), marginalised over lam."""
        lp = logsumexp(self.log_posterior(), axis=1)
        lp -= logsumexp(lp)
        return np.exp(lp)

    def _log_post_at(self, h_values: np.ndarray) -> np.ndarray:
        """Log posterior marginalised over lam at arbitrary thicknesses (for local refinement)."""
        B, C = self._effective()
        a = self.a_ref * np.exp(np.clip(h_values[:, None] / self.lam[None, :], None, 60.0))
        ll = a * B[None, :] - 0.5 * a**2 * C[None, :] + self.log_prior_lam[None, :]
        lph = np.interp(h_values, self.h, self.log_prior_h)
        return logsumexp(ll, axis=1) + lph

    def summary(self) -> dict:
        """Posterior mean, sd, 5/95% quantiles and MAP of the onset thickness."""
        p = self.posterior()
        mean = float(p @ self.h)
        sd = float(np.sqrt(p @ (self.h - mean) ** 2))
        hg = self.h
        step = float(hg[1] - hg[0])
        if sd < 5 * step:
            mode = float(hg[np.argmax(p)])
            half = max(10 * max(sd, step), 1.0)
            hg = np.linspace(max(mode - half, self.h[0]), min(mode + half, self.h[-1]), 801)
            lp = self._log_post_at(hg)
            p = np.exp(lp - logsumexp(lp))
            mean = float(p @ hg)
            sd = float(np.sqrt(p @ (hg - mean) ** 2))
        cdf = np.cumsum(p)
        return {"h_onset_mean": mean, "h_onset_sd": sd,
                "h_onset_q05": float(np.interp(0.05, cdf, hg)),
                "h_onset_q95": float(np.interp(0.95, cdf, hg)),
                "h_onset_map": float(hg[np.argmax(p)])}


class BOCPD:
    """Bayesian online change-point detection, Normal data with unknown mean and unit variance."""

    def __init__(self, *args, **kwargs):
        spec = [
            ("hazard", 1 / 2000),
            ("mu0", 0.0),
            ("var0", 4.0),
            ("recent", 25),
            ("p_fire", 0.9),
            ("max_run", 3000),
        ]
        p = _parse_named_args(args, kwargs, spec)
        self.h = p["hazard"]
        self.mu0, self.var0 = p["mu0"], p["var0"]
        self.recent, self.p_fire, self.max_run = p["recent"], p["p_fire"], p["max_run"]
        self.log_r = np.array([0.0])          # run-length log posterior
        self.mu = np.array([self.mu0])        # posterior mean of segment mean, per run length
        self.var = np.array([self.var0])
        self.t = 0
        self.fired_at_h: float | None = None
        self.change_at_h: float | None = None
        self._h_hist: list[float] = []

    def update(self, h_mm: float, z: float) -> float:
        pred_var = self.var + 1.0
        log_pred = -0.5 * (np.log(2 * np.pi * pred_var) + (z - self.mu) ** 2 / pred_var)
        log_growth = self.log_r + log_pred + np.log1p(-self.h)
        log_cp = logsumexp(self.log_r + log_pred + np.log(self.h))
        new = np.concatenate([[log_cp], log_growth])
        new -= logsumexp(new)
        post_var = 1.0 / (1.0 / self.var + 1.0)
        post_mu = post_var * (self.mu / self.var + z)
        self.mu = np.concatenate([[self.mu0], post_mu])
        self.var = np.concatenate([[self.var0], post_var])
        self.log_r = new
        if len(self.log_r) > self.max_run:
            self.log_r, self.mu, self.var = self.log_r[: self.max_run], self.mu[: self.max_run], self.var[: self.max_run]
            self.log_r -= logsumexp(self.log_r)
        self.t += 1
        self._h_hist.append(h_mm)
        p_recent = float(np.exp(logsumexp(self.log_r[: self.recent + 1])))
        if self.fired_at_h is None and self.t > self.recent * 2 and p_recent > self.p_fire:
            self.fired_at_h = h_mm
            run = int(np.argmax(self.log_r))
            self.change_at_h = self._h_hist[max(len(self._h_hist) - 1 - run, 0)]
        return p_recent


@dataclass
class OnsetResult:
    h_onset_mean: float
    h_onset_sd: float
    h_onset_q05: float
    h_onset_q95: float
    glr_fired_at_h: float | None
    bocpd_fired_at_h: float | None
    bocpd_change_at_h: float | None
    confidence: str
    posterior_h: np.ndarray = field(repr=False, default_factory=lambda: H_GRID)
    posterior: np.ndarray = field(repr=False, default_factory=lambda: np.zeros_like(H_GRID))
    track: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))


def _determine_confidence(g: float | None, b: float | None, agree_mm: float) -> str:
    if g is None and b is None:
        return "none"
    if g is not None and b is not None and abs(g - b) <= agree_mm:
        return "high"
    return "low"


def _feed_detectors(glr: OnsetGLR, bo: BOCPD,
                    feed_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
                    opts: tuple[int, float | None]) -> list:
    h_mm, r, S, H = feed_data
    track_every, stop_after_fire_mm = opts
    track = []
    for k, (hh, rr, ss) in enumerate(zip(h_mm, r, S)):
        glr.update(hh, rr, ss, None if H is None else H[k])
        bo.update(hh, rr / np.sqrt(ss))
        if k % track_every == 0:
            s = glr.summary()
            track.append((hh, s["h_onset_mean"], s["h_onset_sd"]))
        if stop_after_fire_mm is not None and glr.fired_at_h is not None:
            if glr.fired_at_h - hh >= stop_after_fire_mm:
                break
    return track


def detect_onset(*args, **kwargs) -> OnsetResult:
    """Run both detectors over gated, frozen-baseline residuals (in time order)."""
    spec = [
        ("h_mm", None),
        ("r", None),
        ("S", None),
        ("a_ref_N", None),
        ("glr_threshold", 12.0),
        ("agree_mm", 12.0),
        ("h_prior", None),
        ("track_every", 5),
        ("stop_after_fire_mm", None),
        ("bocpd", None),
        ("H", None),
        ("baseline_cov", None),
        ("noise_var", None),
    ]
    p = _parse_named_args(args, kwargs, spec)
    glr = OnsetGLR(p["a_ref_N"], p["glr_threshold"], h_prior=p["h_prior"],
                   baseline_cov=p["baseline_cov"], noise_var=p["noise_var"])
    bo = p["bocpd"] or BOCPD()
    feed_data = (p["h_mm"], p["r"], p["S"], p["H"])
    opts = (p["track_every"], p["stop_after_fire_mm"])
    track = _feed_detectors(glr, bo, feed_data, opts)
    s = glr.summary()
    g, b = glr.fired_at_h, bo.fired_at_h
    conf = _determine_confidence(g, b, p["agree_mm"])
    return OnsetResult(s["h_onset_mean"], s["h_onset_sd"], s["h_onset_q05"], s["h_onset_q95"], g, b,
                       bo.change_at_h, conf, glr.h.copy(), glr.posterior(), np.array(track))


# --------------------------------------------------------------------------- naive baseline

def _evaluate_threshold_peak(gated: np.ndarray, h: np.ndarray, d2: np.ndarray,
                             h_gate_mm: float, k_sigma: float, offset_mm: float) -> tuple[float | None, float | None]:
    ref = gated & (h > h_gate_mm - 40.0)
    ref_vals = d2[ref & np.isfinite(d2)]
    if ref_vals.size < 10:
        return None, None
    med = np.median(ref_vals)
    sd = 1.4826 * np.median(np.abs(ref_vals - med))
    hit = np.flatnonzero(gated & (h < h_gate_mm - 40.0) & (d2 - med > k_sigma * sd))
    if hit.size == 0:
        return None, None
    hd = float(h[hit[0]])
    return hd, hd - offset_mm


def naive_second_derivative(*args, **kwargs) -> tuple[float | None, float | None]:
    """Causal second-derivative threshold. Returns ``(h_detect, h_onset_estimate)``."""
    spec = [
        ("x_mm", None),
        ("force_N", None),
        ("L0_mm", None),
        ("h_gate_mm", None),
        ("h_stop_mm", None),
        ("k_sigma", 6.0),
        ("window_mm", 4.0),
        ("dx_mm", 0.25),
        ("offset_mm", 0.0),
    ]
    p = _parse_named_args(args, kwargs, spec)
    x_mm, force_N, L0_mm = p["x_mm"], p["force_N"], p["L0_mm"]
    h_gate_mm, h_stop_mm = p["h_gate_mm"], p["h_stop_mm"]
    k_sigma, window_mm, dx_mm, offset_mm = p["k_sigma"], p["window_mm"], p["dx_mm"], p["offset_mm"]

    keep = (L0_mm - x_mm) >= h_stop_mm
    c = resample_to_position(x_mm[keep], force_N[keep], dx_mm, x_min=L0_mm - h_gate_mm - 30.0)
    h = L0_mm - c.x_mm
    w = max(int(window_mm / dx_mm) | 1, 7)
    d2 = sg_derivative(c.force_N, w, 2, dx_mm, polyorder=2, causal=True)
    gated = h <= h_gate_mm
    return _evaluate_threshold_peak(gated, h, d2, h_gate_mm, k_sigma, offset_mm)
