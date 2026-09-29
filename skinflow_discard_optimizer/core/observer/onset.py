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


def observable_onset(h_onset_mm: float, lam_mm: float, amp_N: float, a_ref_N: float) -> float:
    """Thickness where ``F_up`` reaches ``a_ref`` for a simulated stroke.

    The simulator places ``h_onset`` where the upturn reaches that billet's own
    (randomised) amplitude. The force can only reveal where it reaches the fixed
    reference amplitude, which is this value. The gap between the two is real
    billet-to-billet noise between the signal and ``h_crit``.
    """
    return h_onset_mm + lam_mm * np.log(amp_N / a_ref_N)


class OnsetGLR:
    """Sequential GLR test and grid posterior over the onset position.

    Two noise models:

    * independent (``baseline_cov=None``): each residual ~ N(0, S_k);
    * marginalised (``baseline_cov`` = the frozen filter's phi covariance P, plus the
      per-update measurement variance ``noise_var`` and each update's regressor row
      ``H``): ``r_k = a*g_k + H_k @ delta + e_k`` with ``delta ~ N(0, P)`` shared by
      *all* residuals and ``e_k ~ N(0, noise_var)``. Treating residuals as
      independent ignores that the frozen baseline's error is common to all of them
      and makes the onset posterior overconfident. Integrating ``delta`` out keeps the
      closed form: ``B`` and ``C`` become ``B - c' A^-1 b`` and ``C - c' A^-1 c`` with
      ``A = P^-1 + sum H H'/R``, ``b = sum H r/R`` and ``c = sum g H/R``.
    """

    def __init__(self, a_ref_N: float, threshold: float = 12.0, lam_grid: np.ndarray = LAM_GRID,
                 h_grid: np.ndarray = H_GRID, h_prior: np.ndarray | None = None,
                 lam_log_prior: np.ndarray | None = None, baseline_cov: np.ndarray | None = None,
                 noise_var: float | None = None):
        self.a_ref = a_ref_N
        self.threshold = threshold
        self.lam = np.asarray(lam_grid, float)
        self.h = np.asarray(h_grid, float)
        nl = len(self.lam)
        self.Bg = np.zeros(nl)
        self.Cgg = np.zeros(nl)
        self.marginal = baseline_cov is not None
        if self.marginal:
            if noise_var is None:
                raise ValueError("marginalised mode needs noise_var")
            self.R = float(noise_var)
            self._scale = np.array([1.0, 1.0, 1e6])       # work in scaled phi (F_tool in MN)
            P = np.asarray(baseline_cov) / np.outer(self._scale, self._scale)
            self.Pinv = np.linalg.inv(P + 1e-15 * np.eye(3))
            self.cgH = np.zeros((nl, 3))
            self.MHH = np.zeros((3, 3))
            self.bH = np.zeros(3)
        self.log_prior_h = np.zeros_like(self.h) if h_prior is None else np.log(np.maximum(h_prior, 1e-300))
        self.log_prior_lam = np.zeros_like(self.lam) if lam_log_prior is None else lam_log_prior
        # a = A_ref * exp(h_on / lam), as a (n_h, n_lam) table
        self._a = self.a_ref * np.exp(np.clip(self.h[:, None] / self.lam[None, :], None, 60.0))
        self.stat = 0.0
        self.fired_at_h: float | None = None
        self.n = 0

    def update(self, h_mm: float, r: float, S: float, H: np.ndarray | None = None) -> float:
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
        """Posterior mean, sd, 5/95% quantiles and MAP of the onset thickness.

        Once the data are informative the posterior is far narrower than the base
        grid step, so it is re-evaluated on a fine local grid around the mode;
        otherwise the quantiles would collapse into one grid cell.
        """
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

    def __init__(self, hazard: float = 1 / 2000, mu0: float = 0.0, var0: float = 4.0,
                 recent: int = 25, p_fire: float = 0.9, max_run: int = 3000):
        self.h = hazard
        self.mu0, self.var0 = mu0, var0
        self.recent, self.p_fire, self.max_run = recent, p_fire, max_run
        self.log_r = np.array([0.0])          # run-length log posterior
        self.mu = np.array([mu0])             # posterior mean of segment mean, per run length
        self.var = np.array([var0])
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
    # posterior summaries tracked as data arrived: (h at update, mean, sd) every few updates
    track: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))


def detect_onset(h_mm: np.ndarray, r: np.ndarray, S: np.ndarray, a_ref_N: float,
                 glr_threshold: float = 12.0, agree_mm: float = 12.0, h_prior: np.ndarray | None = None,
                 track_every: int = 5, stop_after_fire_mm: float | None = None,
                 bocpd: BOCPD | None = None, H: np.ndarray | None = None,
                 baseline_cov: np.ndarray | None = None, noise_var: float | None = None) -> OnsetResult:
    """Run both detectors over gated, frozen-baseline residuals (in time order).

    Pass ``H``, ``baseline_cov`` and ``noise_var`` (from the frozen ``UKFTrace``) to
    marginalise the shared baseline error; that is the default in the pipeline.
    ``stop_after_fire_mm``: stop feeding data this far past the GLR alarm, to mimic a
    decision taken shortly after detection. None uses everything given.
    """
    glr = OnsetGLR(a_ref_N, glr_threshold, h_prior=h_prior, baseline_cov=baseline_cov, noise_var=noise_var)
    bo = bocpd or BOCPD()
    track = []
    for k, (hh, rr, ss) in enumerate(zip(h_mm, r, S)):
        glr.update(hh, rr, ss, None if H is None else H[k])
        bo.update(hh, rr / np.sqrt(ss))
        if k % track_every == 0:
            s = glr.summary()
            track.append((hh, s["h_onset_mean"], s["h_onset_sd"]))
        if stop_after_fire_mm is not None and glr.fired_at_h is not None and glr.fired_at_h - hh >= stop_after_fire_mm:
            break
    s = glr.summary()
    g, b = glr.fired_at_h, bo.fired_at_h
    if g is None and b is None:
        conf = "none"
    elif g is not None and b is not None and abs(g - b) <= agree_mm:
        conf = "high"
    else:
        conf = "low"
    return OnsetResult(s["h_onset_mean"], s["h_onset_sd"], s["h_onset_q05"], s["h_onset_q95"], g, b,
                       bo.change_at_h, conf, glr.h.copy(), glr.posterior(), np.array(track))


# --------------------------------------------------------------------------- naive baseline

def naive_second_derivative(x_mm: np.ndarray, force_N: np.ndarray, L0_mm: float, h_gate_mm: float,
                            h_stop_mm: float, k_sigma: float = 6.0, window_mm: float = 4.0,
                            dx_mm: float = 0.25, offset_mm: float = 0.0) -> tuple[float | None, float | None]:
    """Causal second-derivative threshold. Returns ``(h_detect, h_onset_estimate)``.

    The second derivative of force versus position (causal Savitzky-Golay, past
    samples only) is compared with ``k_sigma`` robust sds of its value over the
    first half of the gated region. The first exceedance gives ``h_detect``, and
    the onset estimate is ``h_detect - offset_mm`` (offset calibrated on training
    strokes).
    """
    keep = (L0_mm - x_mm) >= h_stop_mm
    c = resample_to_position(x_mm[keep], force_N[keep], dx_mm, x_min=L0_mm - h_gate_mm - 30.0)
    h = L0_mm - c.x_mm
    w = max(int(window_mm / dx_mm) | 1, 7)
    d2 = sg_derivative(c.force_N, w, 2, dx_mm, polyorder=2, causal=True)
    gated = h <= h_gate_mm
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
