"""Task 3.3: hierarchical model, cost-optimal cut, conformal calibration."""
import numpy as np
import pytest

from skinflow_discard_optimizer.core.estimator.decision import (
    AdaptiveConformal,
    DecisionSettings,
    chance_constrained_cut,
    cost_optimal_cut,
    expected_cost,
    recommend,
)
from skinflow_discard_optimizer.core.estimator.hierarchical import HierarchicalModel


@pytest.fixture(scope="module")
def st():
    return DecisionSettings.load()


# --- hierarchical model --------------------------------------------------------------------

def _grouped_data(rng, n=1500):
    x = rng.normal(0, 1, (n, 2))
    die = rng.choice(["D1", "D2", "D3", "D4"], n, p=[0.4, 0.3, 0.25, 0.05])
    alloy = rng.choice(["A", "B"], n)
    u = {"D1": -1.0, "D2": 0.5, "D3": 1.5, "D4": -2.0}
    v = {"A": 0.0, "B": 0.8}
    y = 20 + 2.0 * x[:, 0] - 1.0 * x[:, 1] + np.array([u[d] for d in die]) + np.array([v[a] for a in alloy]) \
        + rng.normal(0, 0.7, n)
    return x, y, {"die_id": die, "alloy_id": alloy}


def test_hierarchical_recovers_coefficients_and_effects():
    x, y, g = _grouped_data(np.random.default_rng(0))
    m = HierarchicalModel(["x1", "x2"], ["die_id", "alloy_id"]).fit(x, y, g, iterations=800, burn_in=300)
    coefs = {n: mu for n, mu, _ in m.coef_table()}
    assert coefs["x1"] == pytest.approx(2.0, abs=0.1)
    assert coefs["x2"] == pytest.approx(-1.0, abs=0.1)
    assert m.diagnostics["sigma_mm"] == pytest.approx(0.7, abs=0.08)
    # die effects are only identified up to the shared intercept: check the differences
    eff = dict(zip(m.levels["die_id"], m.effects["die_id"].mean(0) * m.y_sd))
    assert eff["D3"] - eff["D1"] == pytest.approx(2.5, abs=0.3)


def test_unseen_group_gets_wider_prediction():
    x, y, g = _grouped_data(np.random.default_rng(1))
    m = HierarchicalModel(["x1", "x2"], ["die_id", "alloy_id"]).fit(x, y, g, iterations=600, burn_in=200)
    xs = np.zeros((1, 2))
    _, v_seen = m.predict(xs, {"die_id": np.array(["D1"]), "alloy_id": np.array(["A"])})
    _, v_new = m.predict(xs, {"die_id": np.array(["D-NEW"]), "alloy_id": np.array(["A"])})
    assert v_new[0] > v_seen[0]


def test_predictive_is_calibrated_on_held_out_rows():
    rng = np.random.default_rng(2)
    x, y, g = _grouped_data(rng, 2000)
    m = HierarchicalModel(["x1", "x2"], ["die_id", "alloy_id"]).fit(x[:1500], y[:1500],
                                                                    {k: v[:1500] for k, v in g.items()}, 600, 200)
    mu, var = m.predict(x[1500:], {k: v[1500:] for k, v in g.items()})
    z = (y[1500:] - mu) / np.sqrt(var)
    assert np.mean(np.abs(z) < 1.645) == pytest.approx(0.90, abs=0.04)


def test_save_load_roundtrip():
    x, y, g = _grouped_data(np.random.default_rng(3), 400)
    m = HierarchicalModel(["x1", "x2"], ["die_id", "alloy_id"]).fit(x, y, g, 300, 100)
    m2 = HierarchicalModel.from_npz(m.to_npz())
    a = m.predict(x[:5], {k: v[:5] for k, v in g.items()})
    b = m2.predict(x[:5], {k: v[:5] for k, v in g.items()})
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


# --- cost-optimal and chance-constrained cuts -----------------------------------------------

@pytest.mark.parametrize("m,s", [(22.0, 1.0), (25.0, 2.5), (30.0, 0.5)])
def test_closed_form_cut_matches_grid_minimum(st, m, s):
    grid = np.linspace(st.min_cut_mm, st.max_cut_mm, 40001)
    h_grid = grid[np.argmin(expected_cost(grid, m, s, st))]
    assert float(cost_optimal_cut(m, s, st)) == pytest.approx(h_grid, abs=0.01)


def test_cut_grows_with_uncertainty_and_respects_bounds(st):
    cuts = [float(cost_optimal_cut(22.0, s, st)) for s in (0.5, 1.0, 2.0, 4.0)]
    assert np.all(np.diff(cuts) > 0)
    assert float(cost_optimal_cut(80.0, 1.0, st)) == st.max_cut_mm
    assert float(cost_optimal_cut(0.0, 0.5, st)) == st.min_cut_mm


def test_chance_constrained_cut_quantile(st):
    from scipy.stats import norm
    h = float(chance_constrained_cut(25.0, 1.5, st))
    assert norm.sf((h - 25.0) / 1.5) == pytest.approx(st.chance_alpha, rel=1e-6)


# --- conformal -----------------------------------------------------------------------------

def test_split_conformal_hits_target_on_exchangeable_data():
    rng = np.random.default_rng(4)
    cal = AdaptiveConformal(np.abs(rng.standard_t(4, 3000)), target=0.9, gamma=0.0)
    y = rng.standard_t(4, 5000)
    lo, hi = cal.interval(np.zeros(5000), np.ones(5000))
    assert np.mean((y >= lo) & (y <= hi)) == pytest.approx(0.90, abs=0.015)


def test_aci_restores_coverage_after_a_shift():
    rng = np.random.default_rng(5)
    cal = AdaptiveConformal(np.abs(rng.normal(0, 1, 2000)), target=0.9, gamma=0.02)
    covered = []
    for _ in range(4000):                      # the world's noise doubled; the model still says sd 1
        y = rng.normal(0, 2.0)
        lo, hi = cal.interval(0.0, 1.0)
        covered.append(lo <= y <= hi)
        cal.update(y, 0.0, 1.0)
    assert np.mean(covered[-2000:]) == pytest.approx(0.90, abs=0.03)


def test_recommendation_uses_calibrated_scale(st):
    rng = np.random.default_rng(6)
    wide = AdaptiveConformal(np.abs(rng.normal(0, 2.0, 2000)), 0.9, 0.0)   # model sd understates by 2x
    narrow = AdaptiveConformal(np.abs(rng.normal(0, 1.0, 2000)), 0.9, 0.0)
    r_wide, r_ok = recommend(22.0, 1.0, st, wide), recommend(22.0, 1.0, st, narrow)
    assert r_wide.h_crit_sd_cal_mm == pytest.approx(2 * r_ok.h_crit_sd_cal_mm, rel=0.1)
    assert r_wide.h_cut_mm > r_ok.h_cut_mm
    assert r_wide.interval_hi_mm - r_wide.interval_lo_mm > r_ok.interval_hi_mm - r_ok.interval_lo_mm
