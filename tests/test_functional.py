"""Task 2.2: registration and functional PCA."""
import numpy as np
import pandas as pd
import pytest

from skinflow_discard_optimizer.core.functional import (
    FPCA,
    RegistrationGrid,
    choose_k,
    curve_from_stroke,
    registered_curve,
)
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke
from skinflow_discard_optimizer.sim.force_model import Press
from skinflow_discard_optimizer.sim.scenario import Scenario, run_scenario


@pytest.fixture(scope="module")
def press():
    return Press.load()


@pytest.fixture(scope="module")
def healthy_curves(press):
    rows = pd.DataFrame([r for r, _ in run_scenario(Scenario.load("healthy_baseline").with_cycles(160))])
    X = np.array([curve_from_stroke(regenerate_stroke(r, press), press) for r in rows.to_dict("records")])
    return rows, X


def test_registration_aligns_billets_of_different_length():
    g = RegistrationGrid()
    # a curve that depends only on remaining thickness h in the tail, and on u in the body
    def make(L0):
        x = np.linspace(0, L0 - 8, 20000)
        h = L0 - x
        u = (x - g.x_start_mm) / (L0 - g.h_tail_mm - g.x_start_mm)
        f = np.where(h > g.h_tail_mm, 1e6 * (2 - u), 1e6 + 1e4 * (g.h_tail_mm - h))
        return registered_curve(x, f, L0, g)
    a, b = make(790.0), make(812.0)
    assert np.max(np.abs(a - b)) < 0.01 * 1e6


def test_grid_ends_at_max_cut_so_every_cycle_observes_it():
    g = RegistrationGrid()
    assert g.h_tail.min() == pytest.approx(g.h_min_mm)
    assert g.h_min_mm >= 60.0


def test_fpca_orthonormal_and_reconstructs_to_noise_floor(healthy_curves):
    _, X = healthy_curves
    fp = FPCA.fit(X[:120])
    gram = (fp.components * fp.weights) @ fp.components.T
    assert np.allclose(gram, np.eye(fp.k), atol=1e-8)
    scores, spe = fp.project_many(X[120:])
    rms_resid = np.sqrt(spe / fp.weights.sum())
    assert np.median(rms_resid) < 5e3          # kN-level residual on ~8 MN curves
    one = fp.project(X[130])
    assert np.allclose(one.reconstruction + one.residual, X[130])


def test_first_components_have_physical_meaning(healthy_curves):
    rows, X = healthy_curves
    fp = FPCA.fit(X, n_components=3)
    sc, _ = fp.project_many(X)
    r_temp = abs(np.corrcoef(sc[:, 0], rows.true_billet_temp_C)[0, 1])
    # friction explains a large part of the leading components jointly
    A = np.column_stack([sc[:, :3], np.ones(len(sc))])
    coef, *_ = np.linalg.lstsq(A, rows.spec_mu, rcond=None)
    r2_mu = 1 - np.var(rows.spec_mu - A @ coef) / np.var(rows.spec_mu)
    assert r_temp > 0.6 and r2_mu > 0.6


def test_choose_k_one_se_rule():
    cv = np.array([[10, 10.2, 9.8], [3, 3.1, 2.9], [2.0, 2.1, 1.9], [1.98, 2.08, 1.92], [1.97, 2.1, 1.9]])
    assert choose_k(cv) == 2


def test_save_load_roundtrip(healthy_curves, tmp_path):
    _, X = healthy_curves
    fp = FPCA.fit(X, n_components=3)
    fp.save(tmp_path / "f.npz")
    fp2 = FPCA.load(tmp_path / "f.npz")
    assert np.allclose(fp.project(X[0]).scores, fp2.project(X[0]).scores)
