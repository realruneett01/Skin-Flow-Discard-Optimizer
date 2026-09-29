"""Task 1.2: ground-truth butt and defect model."""
import numpy as np
import pytest

from skinflow_discard_optimizer.sim.defect_model import DefectModel, PressState
from skinflow_discard_optimizer.sim.force_model import force_breakdown, simulate_stroke


@pytest.fixture(scope="module")
def model():
    return DefectModel()


def test_h_crit_responds_to_latent_states_with_documented_signs(model):
    s0 = PressState()
    base = model.h_crit_mean(s0)
    assert model.h_crit_mean(PressState(liner_scale_mm=0.6)) > base
    assert model.h_crit_mean(PressState(die_wear=0.8)) > base
    assert model.h_crit_mean(PressState(liner_temp_C=400.0)) > base   # colder container, bigger dT
    assert model.h_crit_mean(PressState(mu_base=0.62)) > base


def test_truth_is_recorded_and_onset_precedes_h_crit(model):
    rng = np.random.default_rng(0)
    truths = [model.sample_truth(PressState(), rng) for _ in range(500)]
    h = np.array([t.h_crit_mm for t in truths])
    on = np.array([t.h_onset_mm for t in truths])
    assert np.all(on >= h)
    assert np.std(h) == pytest.approx(model.p.noise_sd, rel=0.15)
    assert np.mean(on - h) == pytest.approx(model.p.offset_mean, abs=0.2)


def test_defect_probability_smooth_and_monotone(model):
    h_cut = np.linspace(10, 40, 301)
    p = model.defect_probability(h_cut, 25.0)
    assert np.all(np.diff(p) < 0)                       # thicker cut, fewer defects
    assert model.defect_probability(25.0, 25.0) == pytest.approx(0.5)
    assert p[0] > 0.999 and p[-1] < 1e-6


def test_oracle_cut_minimises_true_cost(model):
    for h_crit in [15.0, 22.0, 31.5]:
        grid = np.linspace(model.p.min_cut_mm, model.p.max_cut_mm, 20001)
        cost = model.expected_cost(grid, h_crit)
        h_opt = float(model.oracle_cut(np.array(h_crit)))
        assert h_opt == pytest.approx(grid[np.argmin(cost)], abs=0.01)
        assert model.expected_cost(h_opt, h_crit) <= cost.min() + 1e-9


def test_true_cost_of_any_cut_is_queryable(model):
    rng = np.random.default_rng(3)
    t = model.sample_truth(PressState(), rng)
    h_opt = float(model.oracle_cut(np.array(t.h_crit_mm)))
    c_opt = float(model.expected_cost(h_opt, t.h_crit_mm))
    for h in [h_opt - 3, h_opt + 3, model.p.static_cut_mm]:
        assert model.expected_cost(h, t.h_crit_mm) > c_opt


def test_cost_units(model):
    # 1 mm of discard in a 235 mm bore: ~0.117 kg of aluminium
    assert float(model.discard_mass_kg(1.0)) == pytest.approx(0.1167, rel=0.01)
    assert model.metal_cost_per_mm == pytest.approx(0.1167 * 0.5, rel=0.01)


def test_oracle_margin_positive_and_moderate(model):
    m = model.oracle_margin_mm()
    assert 0 < m < 10


@pytest.mark.parametrize("shape", ["upturn", "drop"])
def test_stroke_carries_truth_and_shape_switch(model, shape):
    rng = np.random.default_rng(5)
    s = PressState(liner_scale_mm=0.4, die_wear=0.5)
    truth = model.sample_truth(s, rng, shape=shape)
    spec = model.stroke_for(s, truth, rng)
    assert spec.h_onset_mm == truth.h_onset_mm and spec.shape == shape
    assert spec.mu == pytest.approx(model.effective_mu(s))
    x = np.array([spec.L0_mm - truth.h_onset_mm - 40, spec.L0_mm - spec.h_end_mm])
    F = force_breakdown(x, spec, model.press).total_N
    assert (F[1] > F[0]) if shape == "upturn" else (F[1] < F[0])
    d = simulate_stroke(spec, model.press, rng)
    assert d.spec.h_onset_mm == truth.h_onset_mm
