"""Curve registration and functional PCA of stroke force curves (Task 2.2, part of layer L7).

Registration
    Billet length varies, so raw stroke position does not line up across cycles. The
    curve is mapped onto a two-part common axis:

    * body: ``x in [x_start, L0 - h_tail]`` mapped linearly to ``u in [0, 1]``
      (absorbs billet length variation);
    * tail: remaining thickness ``h`` from ``h_tail`` down to ``h_min``, in absolute mm,
      because end-of-stroke physics depends on the absolute thickness.

    ``h_min`` defaults to the maximum allowed cut, so the registered curve only
    covers stroke that every cycle records, whatever the cut. The final few mm
    (the onset region) belong to the UKF and onset detector, not FPCA.

FPCA
    Weighted PCA on registered healthy curves, using quadrature weights for the
    non-uniform grid, so eigenfunctions are orthonormal in L2 over the stroke.
    The number of components comes from cross-validation that holds out *grid
    points* of held-out curves: scores are fitted on the visible points and the
    hidden ones predicted. Plain projection error always falls as components are
    added, so it cannot pick a number; this error is U-shaped instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from skinflow_discard_optimizer.core.preprocess import resample_to_position
from skinflow_discard_optimizer.sim.force_model import Press, StrokeData, pressures_to_force


@dataclass(frozen=True)
class RegistrationGrid:
    x_start_mm: float = 2.0
    h_tail_mm: float = 120.0
    h_min_mm: float = 60.0
    n_body: int = 300
    tail_step_mm: float = 1.0

    @property
    def u_body(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_body)

    @property
    def h_tail(self) -> np.ndarray:
        n = int(round((self.h_tail_mm - self.h_min_mm) / self.tail_step_mm))
        return self.h_tail_mm - self.tail_step_mm * np.arange(1, n + 1)

    @property
    def size(self) -> int:
        return self.n_body + len(self.h_tail)

    def x_of(self, L0_mm: float) -> np.ndarray:
        """Stroke positions (mm) of the grid points for a billet with upset length L0."""
        body = self.x_start_mm + self.u_body * (L0_mm - self.h_tail_mm - self.x_start_mm)
        return np.concatenate([body, L0_mm - self.h_tail])

    def weights(self, L0_ref: float = 800.0) -> np.ndarray:
        """Trapezoid quadrature weights in mm for a reference billet."""
        x = self.x_of(L0_ref)
        w = np.zeros_like(x)
        dx = np.diff(x)
        w[:-1] += dx / 2
        w[1:] += dx / 2
        return w

    def axis_label(self) -> np.ndarray:
        """Plot axis: body as 0..(L0_ref-h_tail) mm, tail continuing in mm."""
        return self.x_of(800.0)


def registered_curve(x_mm: np.ndarray, force_N: np.ndarray, L0_mm: float,
                     grid: RegistrationGrid = RegistrationGrid(), **kwargs) -> np.ndarray:
    """Register one stroke (any x ordering) onto the common grid, averaging over each grid cell."""
    dx_mm: float = kwargs.get("dx_mm", 0.1)
    c = resample_to_position(np.asarray(x_mm), np.asarray(force_N), dx_mm, x_min=0.0)
    xg = grid.x_of(L0_mm)
    spacing = np.gradient(xg)
    out = np.empty(grid.size)
    # cell-average: smooth at body spacing on the body, tail spacing on the tail
    for sl, step in ((slice(0, grid.n_body), float(np.median(spacing[:grid.n_body]))),
                     (slice(grid.n_body, None), grid.tail_step_mm)):
        size = max(int(round(step / dx_mm)), 1)
        sm = uniform_filter1d(c.force_N, size=size, mode="nearest")
        out[sl] = np.interp(xg[sl], c.x_mm, sm)
    return out


def curve_from_stroke(d: StrokeData, press: Press, L0_mm: float | None = None,
                      grid: RegistrationGrid = RegistrationGrid()) -> np.ndarray:
    """Registered force curve from measured cap/rod pressure and measured position."""
    L0 = d.spec.L0_mm if L0_mm is None else L0_mm
    return registered_curve(d.x_mm, pressures_to_force(d.p_cap_bar, d.p_rod_bar, press), L0, grid)


# --------------------------------------------------------------------------- FPCA


@dataclass
class Projection:
    scores: np.ndarray
    reconstruction: np.ndarray
    spe: float                  # squared prediction error (weighted L2^2, N^2 * mm)
    residual: np.ndarray


def _fit_fpca(cls, curves: np.ndarray, grid: RegistrationGrid = RegistrationGrid(),
              **kwargs) -> "FPCA":
    n_components: int | None = kwargs.get("n_components", None)
    max_components: int = kwargs.get("max_components", 12)
    cv_folds: int = kwargs.get("cv_folds", 5)
    rng: np.random.Generator | None = kwargs.get("rng", None)

    curves = np.asarray(curves, dtype=float)
    w = grid.weights()
    sw = np.sqrt(w)
    mean = curves.mean(axis=0)
    Z = (curves - mean) * sw
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    lam = s**2 / (len(curves) - 1)
    cv = None
    if n_components is None:
        cv = cv_missing_points(curves, grid, max_components, cv_folds, rng=rng)
        n_components = choose_k(cv)
    comps = vt[:n_components] / sw          # back to function space: sum_j w_j phi_a phi_b = delta
    scores = (curves - mean) @ (comps * w).T
    return cls(grid, mean, comps, lam[:n_components], np.cov(scores, rowvar=False).reshape(n_components, n_components),
               w, lam[:n_components] / lam.sum(), cv, len(curves))


def _load_fpca(cls, path: Path) -> "FPCA":
    z = np.load(path)
    a, b, c, n, s = z["grid"]
    grid = RegistrationGrid(float(a), float(b), float(c), int(n), float(s))
    cv = z["cv_errors"]
    return cls(grid, z["mean"], z["components"], z["eigenvalues"], z["score_cov"], z["weights"],
               z["explained_ratio"], cv if cv.size else None, int(z["n_train"]))


@dataclass
class FPCA:
    grid: RegistrationGrid
    mean: np.ndarray
    components: np.ndarray      # (K, P) eigenfunctions, orthonormal in weighted L2
    eigenvalues: np.ndarray     # (K,)
    score_cov: np.ndarray       # (K, K) empirical covariance of training scores
    weights: np.ndarray
    explained_ratio: np.ndarray  # per component, of total variance
    cv_errors: np.ndarray | None = None
    n_train: int = 0

    fit = classmethod(_fit_fpca)
    load = classmethod(_load_fpca)

    @property
    def k(self) -> int:
        return len(self.eigenvalues)

    def project_many(self, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scores (N, K) and SPE (N,) for a batch of registered curves."""
        c = np.asarray(curves, dtype=float)
        scores = (c - self.mean) @ (self.components * self.weights).T
        resid = c - (self.mean + scores @ self.components)
        assert scores.shape[-1] == self.k
        return scores, np.sum(self.weights * resid**2, axis=-1)

    def project(self, curve: np.ndarray) -> Projection:
        scores, spe = self.project_many(curve[None, :])
        recon = self.mean + scores[0] @ self.components
        resid = np.asarray(curve, dtype=float) - recon
        return Projection(scores[0], recon, float(spe[0]), resid)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        g = self.grid
        np.savez(path, mean=self.mean, components=self.components, eigenvalues=self.eigenvalues,
                 score_cov=self.score_cov, weights=self.weights, explained_ratio=self.explained_ratio,
                 cv_errors=np.array([]) if self.cv_errors is None else self.cv_errors,
                 n_train=self.n_train,
                 grid=np.array([g.x_start_mm, g.h_tail_mm, g.h_min_mm, g.n_body, g.tail_step_mm]))


def _predict_missing_at_k(k: int, phi: np.ndarray, r_vis: np.ndarray, vis: np.ndarray, hide: np.ndarray) -> np.ndarray:
    if k == 0:
        return np.zeros(hide.sum())
    A = phi[:k, vis].T
    coef, *_ = np.linalg.lstsq(A, r_vis, rcond=None)
    return coef @ phi[:k, hide]


def _eval_curve_cv(c: np.ndarray, mean: np.ndarray, phi: np.ndarray, max_k: int,
                   rng: np.random.Generator, hide_frac: float = 0.2) -> np.ndarray:
    p = len(c)
    hide = rng.random(p) < hide_frac
    vis = ~hide
    r = c - mean
    r_vis = r[vis]
    errs = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        pred = _predict_missing_at_k(k, phi, r_vis, vis, hide)
        errs[k] = np.mean((r[hide] - pred) ** 2)
    return errs


def cv_missing_points(curves: np.ndarray, grid: RegistrationGrid, max_k: int, folds: int = 5,
                      **kwargs) -> np.ndarray:
    """Cross-validated prediction error of hidden grid points, for K = 0..max_k.

    For each fold: fit mean and eigenfunctions on the other curves; for each held-out
    curve, hide ``hide_frac`` of its grid points, fit K scores by least squares on
    the visible points, and predict the hidden ones. Returns a ``(max_k+1, folds)``
    array of mean squared prediction errors (N^2).
    """
    rng: np.random.Generator = kwargs.get("rng") or np.random.default_rng(0)
    hide_frac: float = kwargs.get("hide_frac", 0.2)
    n, _ = curves.shape
    w = grid.weights()
    sw = np.sqrt(w)
    fold_of = rng.permutation(n) % folds
    errs = np.zeros((max_k + 1, folds))
    for f in range(folds):
        tr, te = curves[fold_of != f], curves[fold_of == f]
        mean = tr.mean(axis=0)
        _, _, vt = np.linalg.svd((tr - mean) * sw, full_matrices=False)
        phi = vt[:max_k] / sw
        for c in te:
            errs[:, f] += _eval_curve_cv(c, mean, phi, max_k, rng, hide_frac) / len(te)
    return errs


def choose_k(cv: np.ndarray) -> int:
    """Smallest K whose mean CV error is within one standard error of the minimum (1-SE rule)."""
    m = cv.mean(axis=1)
    se = cv.std(axis=1, ddof=1) / np.sqrt(cv.shape[1])
    best = int(np.argmin(m))
    ok = np.flatnonzero(m <= m[best] + se[best])
    return max(int(ok[0]), 1)
