"""Randomised checks of the numeric helpers (seeded, so every failure is reproducible)."""
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_lse
from scipy.stats import norm

from skinflow_discard_optimizer.core.estimator.decision import (
    AdaptiveConformal,
    DecisionSettings,
    _isf,
    _sf,
    cost_optimal_cut,
    expected_cost,
)
from skinflow_discard_optimizer.core.observer.onset import logsumexp


@pytest.mark.parametrize("seed", range(600))
def test_logsumexp_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(n) for n in rng.integers(1, 7, size=rng.integers(1, 4)))
    a = rng.normal(0.0, float(rng.choice([1.0, 30.0, 700.0])), shape)
    special = seed % 6
    if special == 1:
        a.flat[0] = -np.inf
    elif special == 2:
        a[...] = -np.inf
    elif special == 3:
        a.flat[-1] = np.nan
    elif special == 4:
        a.flat[0] = np.inf
    elif special == 5:
        a[...] = a.flat[0]                       # ties
    for axis in [None, *range(len(shape))]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref, got = scipy_lse(a, axis=axis), logsumexp(a, axis=axis)
        assert np.shape(ref) == np.shape(got)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("seed", range(200))
def test_isf_sf_match_scipy(seed):
    rng = np.random.default_rng(10_000 + seed)
    q = np.concatenate([rng.uniform(0, 1, 50), 10.0 ** rng.uniform(-300, 0, 20), [0.0, 1.0, 0.5]])
    z = np.concatenate([rng.normal(0, rng.choice([1.0, 5.0, 20.0]), 50), [-np.inf, np.inf, 0.0]])
    np.testing.assert_allclose(_isf(q), norm.isf(q), rtol=1e-12, atol=0)
    np.testing.assert_allclose(_sf(z), norm.sf(z), rtol=1e-12, atol=0)


@pytest.fixture(scope="module")
def settings():
    return DecisionSettings.load()


@pytest.mark.parametrize("seed", range(300))
def test_cost_optimal_cut_is_bounded_and_locally_optimal(seed, settings):
    rng = np.random.default_rng(20_000 + seed)
    m = float(rng.uniform(-30, 120))
    s = float(10.0 ** rng.uniform(-2, 2))
    h = float(cost_optimal_cut(m, s, settings))
    assert np.isfinite(h) and settings.min_cut_mm <= h <= settings.max_cut_mm
    if h in (settings.min_cut_mm, settings.max_cut_mm):
        return                                   # clipped to a safety bound: optimality is not claimed there
    c0 = float(expected_cost(h, m, s, settings))
    for d in (-0.5, 0.5):
        h2 = h + d
        if settings.min_cut_mm <= h2 <= settings.max_cut_mm:
            assert c0 <= float(expected_cost(h2, m, s, settings)) + 1e-9, (m, s, h)


@pytest.mark.parametrize("seed", range(100))
def test_conformal_is_well_behaved_over_random_streams(seed):
    rng = np.random.default_rng(30_000 + seed)
    scores = np.abs(rng.normal(0, rng.uniform(0.3, 3.0), int(rng.integers(50, 400))))
    cal = AdaptiveConformal(scores)
    for _ in range(60):
        m, s = float(rng.uniform(10, 40)), float(rng.uniform(0.2, 5))
        lo, hi = cal.interval(m, s)
        assert lo <= m <= hi and np.isfinite(lo) and np.isfinite(hi)
        assert cal.q() >= 0.0 and np.isfinite(cal.q())
        assert isinstance(cal.update(float(m + rng.normal(0, 2 * s)), m, s), (bool, np.bool_))
    snap = cal.snapshot()
    assert snap.q() == pytest.approx(cal.q())
