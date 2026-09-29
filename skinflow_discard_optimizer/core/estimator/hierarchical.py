"""Hierarchical Bayesian linear regression with partial pooling (Task 3.3, layers L5/L6).

Part of the Bayesian Discard Thickness Estimator::

    y_i = x_i' beta + u[die_i] + v[alloy_i] + e_i
    e_i ~ N(0, sigma^2),   u_j ~ N(0, tau_u^2),   v_k ~ N(0, tau_v^2)
    beta ~ N(0, 10^2 I) on standardised inputs;  sigma^2, tau^2 ~ InvGamma(2, 1) on standardised y

Fitted by Gibbs sampling. Every conditional is conjugate, so the sampler is exact
and short. The plan names PyMC or NumPyro, but neither supports Python 3.14 here
(docs/assumptions.md A-25), and a hand-written conjugate sampler is fully
transparent.

Partial pooling: a die seen many times gets its own intercept; a die seen a few
times is pulled toward the population; an unseen die gets u ~ N(0, tau_u^2), so
its predictions are honestly wider. The same holds for alloys.

``predict`` returns the Gaussian approximation of the posterior predictive (mean
and variance, integrating over parameter uncertainty).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HierarchicalModel:
    feature_names: list[str]
    group_names: list[str]                       # e.g. ["die_id", "alloy_id"]
    x_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_sd: np.ndarray = field(default_factory=lambda: np.ones(0))
    y_mean: float = 0.0
    y_sd: float = 1.0
    beta: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))          # draws (D, p+1) incl. intercept
    sigma2: np.ndarray = field(default_factory=lambda: np.zeros(0))             # draws (D,)
    levels: dict[str, list[str]] = field(default_factory=dict)
    effects: dict[str, np.ndarray] = field(default_factory=dict)                # group -> draws (D, n_levels)
    tau2: dict[str, np.ndarray] = field(default_factory=dict)                   # group -> draws (D,)
    diagnostics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y: np.ndarray, groups: dict[str, np.ndarray], iterations: int = 1500,
            burn_in: int = 500, seed: int = 0) -> "HierarchicalModel":
        rng = np.random.default_rng(seed)
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.x_mean, self.x_sd = X.mean(0), X.std(0) + 1e-12
        self.y_mean, self.y_sd = float(y.mean()), float(y.std() + 1e-12)
        Z = np.column_stack([np.ones(len(X)), (X - self.x_mean) / self.x_sd])
        ys = (y - self.y_mean) / self.y_sd
        n, p = Z.shape

        idx, counts = {}, {}
        for g in self.group_names:
            lv, inv = np.unique(np.asarray(groups[g]).astype(str), return_inverse=True)
            self.levels[g] = lv.tolist()
            idx[g] = inv
            counts[g] = np.bincount(inv, minlength=len(lv)).astype(float)
        eff = {g: np.zeros(len(self.levels[g])) for g in self.group_names}
        tau2 = {g: 0.1 for g in self.group_names}
        sigma2 = 0.5
        ZtZ = Z.T @ Z
        prior_prec = np.eye(p) / 100.0
        a0, b0 = 2.0, 1.0

        keep = iterations - burn_in
        B = np.empty((keep, p))
        S2 = np.empty(keep)
        E = {g: np.empty((keep, len(self.levels[g]))) for g in self.group_names}
        T2 = {g: np.empty(keep) for g in self.group_names}
        for it in range(iterations):
            offset = sum(eff[g][idx[g]] for g in self.group_names)
            # beta | rest
            prec = ZtZ / sigma2 + prior_prec
            cov = np.linalg.inv(prec)
            mean = cov @ (Z.T @ (ys - offset) / sigma2)
            beta = rng.multivariate_normal(mean, cov)
            lin = Z @ beta
            # random effects | rest, one group at a time
            for g in self.group_names:
                other = sum(eff[h][idx[h]] for h in self.group_names if h != g)
                r = ys - lin - other
                sums = np.bincount(idx[g], weights=r, minlength=len(counts[g]))
                v = 1.0 / (counts[g] / sigma2 + 1.0 / tau2[g])
                eff[g] = rng.normal(v * sums / sigma2, np.sqrt(v))
                tau2[g] = 1.0 / rng.gamma(a0 + len(eff[g]) / 2, 1.0 / (b0 + 0.5 * eff[g] @ eff[g]))
            resid = ys - lin - sum(eff[g][idx[g]] for g in self.group_names)
            sigma2 = 1.0 / rng.gamma(a0 + n / 2, 1.0 / (b0 + 0.5 * resid @ resid))
            if it >= burn_in:
                k = it - burn_in
                B[k], S2[k] = beta, sigma2
                for g in self.group_names:
                    E[g][k], T2[g][k] = eff[g], tau2[g]
        self.beta, self.sigma2, self.effects, self.tau2 = B, S2, E, T2
        self.diagnostics = {"n": n, "sigma_mm": float(np.sqrt(S2.mean()) * self.y_sd),
                            "geweke_z_sigma": _geweke(S2), "geweke_z_beta0": _geweke(B[:, 0])}
        return self

    # ------------------------------------------------------------------ prediction
    def predict(self, X: np.ndarray, groups: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior predictive mean and variance of y (original units) for each row."""
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.column_stack([np.ones(len(X)), (X - self.x_mean) / self.x_sd])
        lin = Z @ self.beta.T                                     # (n, D)
        eff_mean_draw = np.zeros_like(lin)
        extra_var = np.zeros(len(X))
        for g in self.group_names:
            lv = {name: i for i, name in enumerate(self.levels[g])}
            cols = np.array([lv.get(str(v), -1) for v in np.asarray(groups[g])])
            known = cols >= 0
            eff_mean_draw[known] += self.effects[g][:, cols[known]].T
            extra_var[~known] += float(self.tau2[g].mean())      # unseen level: fresh draw from the population
        draws = lin + eff_mean_draw
        mean = draws.mean(axis=1)
        var = draws.var(axis=1) + float(self.sigma2.mean()) + extra_var
        return mean * self.y_sd + self.y_mean, var * self.y_sd**2

    def coef_table(self) -> list[tuple[str, float, float]]:
        """Posterior mean and sd of each coefficient in original units (per unit of the feature)."""
        out = [("intercept", float(self.beta[:, 0].mean() * self.y_sd + self.y_mean), float(self.beta[:, 0].std() * self.y_sd))]
        for j, nm in enumerate(self.feature_names):
            b = self.beta[:, j + 1] * self.y_sd / self.x_sd[j]
            out.append((nm, float(b.mean()), float(b.std())))
        return out

    # ------------------------------------------------------------------ persistence
    def to_npz(self) -> dict:
        d = {"feature_names": np.array(self.feature_names), "group_names": np.array(self.group_names),
             "x_mean": self.x_mean, "x_sd": self.x_sd, "y": np.array([self.y_mean, self.y_sd]),
             "beta": self.beta, "sigma2": self.sigma2}
        for g in self.group_names:
            d[f"levels__{g}"] = np.array(self.levels[g])
            d[f"effects__{g}"] = self.effects[g]
            d[f"tau2__{g}"] = self.tau2[g]
        return d

    @classmethod
    def from_npz(cls, z) -> "HierarchicalModel":
        m = cls([str(s) for s in z["feature_names"]], [str(s) for s in z["group_names"]])
        m.x_mean, m.x_sd = z["x_mean"], z["x_sd"]
        m.y_mean, m.y_sd = float(z["y"][0]), float(z["y"][1])
        m.beta, m.sigma2 = z["beta"], z["sigma2"]
        for g in m.group_names:
            m.levels[g] = [str(s) for s in z[f"levels__{g}"]]
            m.effects[g] = z[f"effects__{g}"]
            m.tau2[g] = z[f"tau2__{g}"]
        # diagnostics are not stored; they follow from the saved draws
        m.diagnostics = {"n": None, "sigma_mm": float(np.sqrt(m.sigma2.mean()) * m.y_sd),
                         "geweke_z_sigma": _geweke(m.sigma2), "geweke_z_beta0": _geweke(m.beta[:, 0])}
        return m


def _geweke(chain: np.ndarray) -> float:
    """Geweke z-score: mean of the first 10% vs the last 50% of a chain (|z| < 2 suggests convergence)."""
    n = len(chain)
    a, b = chain[: n // 10], chain[n // 2:]
    return float((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + 1e-300))
