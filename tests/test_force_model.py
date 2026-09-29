"""Task 1.1: forward force-curve model."""
import numpy as np
import pytest

from skinflow_discard_optimizer.sim.force_model import (
    KELVIN,
    Alloy,
    Press,
    base_force_N,
    feltham_strain_rate,
    flow_stress_MPa,
    force_breakdown,
    force_to_pressures,
    nominal_stroke,
    pressures_to_force,
    simulate_stroke,
    upturn_coefficient,
    zener_hollomon,
)


@pytest.fixture(scope="module")
def press():
    return Press.load()


@pytest.fixture(scope="module", params=["AA6063", "AA6082"])
def alloy(request):
    return Alloy.from_config(request.param)


# --- dimensional consistency -------------------------------------------------------

def test_strain_rate_scales_as_speed_over_length():
    base = feltham_strain_rate(10.0, 235.0, 40.0)
    assert feltham_strain_rate(20.0, 235.0, 40.0) == pytest.approx(2 * base)
    # all lengths x k, speed unchanged -> rate / k   (units 1/s)
    assert feltham_strain_rate(10.0, 2 * 235.0, 40.0) == pytest.approx(base / 2)
    # mm -> m with speed in m/s gives the same number
    assert feltham_strain_rate(10.0e-3, 0.235, 40.0) == pytest.approx(base)


def test_strain_rate_matches_hand_calculation():
    Db, R, v = 235.0, 40.0, 12.0
    De = Db / np.sqrt(R)
    expected = 6 * v * Db**2 * np.log(R) / (Db**3 - De**3)
    assert feltham_strain_rate(v, Db, R) == pytest.approx(expected)
    assert 0.3 < expected < 5.0  # typical extrusion mean strain rates


def test_zener_hollomon_log_space_consistent(alloy):
    eps, T = 1.3, 760.0
    Z = zener_hollomon(eps, T, alloy.Q)
    direct = np.arcsinh((Z / np.exp(alloy.lnA)) ** (1 / alloy.n)) / alloy.alpha
    assert flow_stress_MPa(eps, T, alloy) == pytest.approx(direct)


def test_force_units_mpa_times_mm2_is_newton():
    # 1 MPa on 1000 mm^2 with ln R = 1 and no friction -> 1000 N
    Db = np.sqrt(4 * 1000 / np.pi)
    F = base_force_N(x_mm=0.0, sigma_MPa=1.0, spec_or_L0=0.0, Db_mm=Db, R=np.e, mu=0.0)
    assert F == pytest.approx(1000.0)


def test_pressure_force_round_trip(press):
    F = np.array([0.0, 5e6, 20e6])
    p_cap, p_rod = force_to_pressures(F, press)
    assert pressures_to_force(p_cap, p_rod, press) == pytest.approx(F)
    # 28 MN rated force must be reachable below 400 bar (contract range)
    p28, _ = force_to_pressures(press.rated_force_N, press)
    assert p28 < 400.0


# --- physical behaviour --------------------------------------------------------------

def test_flow_stress_decreases_monotonically_with_temperature(alloy):
    T = np.linspace(380, 560, 200) + KELVIN
    s = flow_stress_MPa(1.0, T, alloy)
    assert np.all(np.diff(s) < 0)


def test_flow_stress_increases_with_strain_rate(alloy):
    eps = np.logspace(-2, 2, 100)
    s = flow_stress_MPa(eps, 480 + KELVIN, alloy)
    assert np.all(np.diff(s) > 0)


def test_flow_stress_plausible_magnitude(alloy):
    s = flow_stress_MPa(1.0, 480 + KELVIN, alloy)
    assert 10.0 < s < 60.0


def test_6082_stronger_than_6063():
    a, b = Alloy.from_config("AA6063"), Alloy.from_config("AA6082")
    assert flow_stress_MPa(1.0, 753.0, b) > flow_stress_MPa(1.0, 753.0, a)


def test_friction_force_falls_along_stroke(press):
    spec = nominal_stroke(press=press, heating_K=0.0, taper_K=0.0)
    x = np.linspace(100, spec.L0_mm - 100, 50)
    br = force_breakdown(x, spec, press)
    assert np.all(np.diff(br.base_N) < 0)


def test_upturn_coefficient_hits_amplitude_at_onset(press):
    spec = nominal_stroke(press=press)
    a = upturn_coefficient(spec.upturn_amp_N, spec.h_onset_mm, spec.lam_mm)
    assert a * np.exp(-spec.h_onset_mm / spec.lam_mm) == pytest.approx(spec.upturn_amp_N)


def test_upturn_and_drop_shapes(press):
    up = nominal_stroke(press=press, shape="upturn")
    dn = nominal_stroke(press=press, shape="drop")
    x = np.array([up.L0_mm - 100.0, up.L0_mm - up.h_end_mm])
    f_up = force_breakdown(x, up, press).total_N
    f_dn = force_breakdown(x, dn, press).total_N
    assert f_up[1] > f_up[0]   # rises at the end
    assert f_dn[1] < f_dn[0]   # falls at the end
    # identical well before onset
    assert f_up[0] == pytest.approx(f_dn[0], rel=1e-6)


# --- stroke simulation ----------------------------------------------------------------

def test_simulated_stroke_sampling_and_shape(press):
    spec = nominal_stroke(press=press)
    d = simulate_stroke(spec, press, np.random.default_rng(1))
    assert np.allclose(np.diff(d.t_s), 1e-3)                       # 1 kHz
    assert d.x_true_mm[-1] == pytest.approx(spec.stroke_length_mm, abs=0.1)
    assert np.all(np.diff(d.x_true_mm) >= 0)
    F = d.force_true_N
    i_peak = np.argmax(F[: len(F) // 5])
    assert d.x_true_mm[i_peak] < 0.05 * spec.L0_mm                 # breakthrough early
    assert F.max() < press.rated_force_N
    # noise level ~ pressure noise * cap area
    resid = d.force_measured_N(press) - F
    expected_sd = press.pressure_noise_bar / 10 * np.hypot(press.cap_area_mm2, press.rod_area_mm2)
    assert np.std(resid) == pytest.approx(expected_sd, rel=0.05)


def test_ram_stalls_at_supply_pressure_limit(press):
    from skinflow_discard_optimizer.sim.force_model import stall_limit_N
    spec = nominal_stroke(press=press, h_onset_mm=45.0, lam_mm=5.0)   # upturn far beyond press capacity
    d = simulate_stroke(spec, press, noise=False)
    assert d.force_true_N.max() <= stall_limit_N(press)
    assert d.h_true_mm[-1] > spec.h_end_mm + 1.0                       # stroke ended early
    assert d.p_cap_bar.max() <= press.supply_pressure_nominal_bar + 1e-6


def test_simulation_is_reproducible(press):
    spec = nominal_stroke(press=press)
    a = simulate_stroke(spec, press, np.random.default_rng(7))
    b = simulate_stroke(spec, press, np.random.default_rng(7))
    assert np.array_equal(a.p_cap_bar, b.p_cap_bar)


def test_pressures_within_contract(press):
    from skinflow_discard_optimizer.contracts.validator import check_array
    d = simulate_stroke(nominal_stroke("AA6082", press=press, ram_speed_mm_s=20.0,
                                       T_front_C=430.0), press, np.random.default_rng(2))
    assert check_array("ram_cap_pressure", d.p_cap_bar).all()
    assert check_array("ram_position", d.x_mm).all()
