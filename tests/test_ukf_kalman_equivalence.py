"""``UKF.update`` uses the closed-form Kalman update; it must equal the sigma-point (unscented) update it replaced."""
import numpy as np
import pytest

from skinflow_discard_optimizer.core.observer.ukf import PHI_SCALE, UKF, UKFParams


def unscented_reference(ukf: UKF, y: float, H: np.ndarray, r_var: float):
    L = np.linalg.cholesky(ukf.c * ukf.P)                      # no jitter: the exact unscented update
    X = np.vstack([ukf.x, ukf.x + L.T, ukf.x - L.T])
    Y = X @ (H * PHI_SCALE)
    y_hat = ukf.wm @ Y
    dY = Y - y_hat
    S = float(ukf.wc @ (dY * dY)) + r_var
    C = (ukf.wc * dY) @ (X - ukf.x)
    K = C / S
    nu = float(y - y_hat)
    x = ukf.x + K * nu
    P = ukf.P - np.outer(K, K) * S
    return nu, S, x, 0.5 * (P + P.T)


@pytest.mark.parametrize("seed", range(300))
def test_kalman_update_equals_unscented_update(seed):
    rng = np.random.default_rng(900_000 + seed)
    theta0 = np.array([rng.uniform(0.7, 1.4), rng.uniform(0.2, 0.9), rng.uniform(2e5, 9e5)])
    sd = theta0 * rng.uniform(0.02, 0.4, 3)
    params = UKFParams(q_per_mm=np.array([1e-7, 1e-7, 1e5]) * rng.uniform(0.1, 10))
    ukf = UKF(theta0, np.diag(sd**2), params)
    for _ in range(int(rng.integers(1, 25))):
        H = np.array([rng.uniform(1e6, 2e8), rng.uniform(1e6, 2e8), 1.0])
        y = float(rng.uniform(1e7, 3e7))
        r_var = float(rng.uniform(1e6, 1e9))
        ukf.predict(float(rng.uniform(0, 5)))
        nu_ref, S_ref, x_ref, P_ref = unscented_reference(ukf, y, H, r_var)
        nu, S = ukf.update(y, H, r_var)
        assert nu == pytest.approx(nu_ref, rel=1e-9, abs=1e-4)
        assert S == pytest.approx(S_ref, rel=1e-9)
        np.testing.assert_allclose(ukf.x, x_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(ukf.P, P_ref, rtol=1e-6, atol=1e-12)
