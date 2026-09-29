"""Unit tests for Task 3.4 joint objective (discard cut coupled with dead-cycle time and energy)."""
import numpy as np
import pytest

from skinflow_discard_optimizer.core.joint import (
    JointSettings,
    cost_breakdown,
    cycle_time_delta_s,
    energy_delta_kwh,
    generate_report,
    joint_expected_cost,
    joint_optimal_cut,
    joint_optimal_cut_numerical,
    sensitivity_defect_loss,
    sensitivity_electricity_tariff,
    sensitivity_metal_price,
    sensitivity_throughput_value,
    single_objective_cut,
)


@pytest.fixture
def st() -> JointSettings:
    return JointSettings.load()


def test_joint_settings_slopes(st: JointSettings):
    # Ram speed is 12 mm/s => 1/12 ~ 0.0833 s/mm
    # Shear is 0.004 s/mm
    # Cycle time slope = 0.004 - 0.0833 ~ -0.0793 s/mm
    assert st.cycle_time_slope_s_per_mm == pytest.approx(0.004 - (1.0 / 12.0), abs=1e-5)
    # Energy slope is negative (cutting thicker saves stroke and energy)
    assert st.energy_slope_kwh_per_mm < 0
    # Effective metal cost is less than gross metal cost because extruding longer has throughput cost
    assert st.effective_metal_cost_per_mm < st.metal_cost_per_mm
    assert st.effective_metal_cost_per_mm > 0


def test_deltas_at_reference_cut(st: JointSettings):
    h_ref = st.reference_cut_mm
    assert float(cycle_time_delta_s(h_ref, st)) == pytest.approx(0.0, abs=1e-9)
    assert float(energy_delta_kwh(h_ref, st)) == pytest.approx(0.0, abs=1e-9)

    # Thinner cut takes more extrusion time and more electrical energy
    assert float(cycle_time_delta_s(h_ref - 5.0, st)) > 0
    assert float(energy_delta_kwh(h_ref - 5.0, st)) > 0


def test_cost_breakdown_consistency(st: JointSettings):
    m, s = 25.0, 1.5
    h = 28.0
    bd = cost_breakdown(h, m, s, st)
    expected_sum = bd["metal_cost"] + bd["defect_cost"] + bd["throughput_cost"] + bd["energy_cost"]
    assert float(bd["total_cost"]) == pytest.approx(float(expected_sum), abs=1e-6)
    assert float(joint_expected_cost(h, m, s, st)) == pytest.approx(float(expected_sum), abs=1e-6)


def test_closed_form_matches_numerical_optimization(st: JointSettings):
    # Test across multiple posterior scenarios
    for m in [20.0, 24.0, 28.0, 32.0]:
        for s in [0.8, 1.3, 2.0]:
            h_closed = float(joint_optimal_cut(m, s, st))
            h_num = joint_optimal_cut_numerical(m, s, st)
            assert h_closed == pytest.approx(h_num, abs=1e-3)


def test_joint_matches_single_when_coupling_zeroed(st: JointSettings):
    st_uncoupled = st.with_overrides(
        throughput_value_eur_s=0.0,
        tariff_eur_kwh=0.0,
    )
    m, s = 25.0, 1.5
    h_single = float(single_objective_cut(m, s, st_uncoupled))
    h_joint = float(joint_optimal_cut(m, s, st_uncoupled))
    assert h_joint == pytest.approx(h_single, abs=1e-6)


def test_joint_cut_is_slightly_thicker_than_single(st: JointSettings):
    # Because extruding thinner incurs throughput cost (time is money), joint cut is thicker
    m, s = 24.0, 1.3
    h_single = float(single_objective_cut(m, s, st))
    h_joint = float(joint_optimal_cut(m, s, st))
    assert h_joint > h_single
    assert 0.05 < (h_joint - h_single) < 1.0


def test_sensitivity_analyses(st: JointSettings):
    m, s = 24.0, 1.3
    df_metal = sensitivity_metal_price(m, s, st)
    assert not df_metal.empty
    # As metal price rises, optimal cut shifts thinner (smaller h)
    assert df_metal.h_joint_mm.iloc[-1] < df_metal.h_joint_mm.iloc[0]

    df_defect = sensitivity_defect_loss(m, s, st)
    assert not df_defect.empty
    # As defect loss rises, optimal cut shifts thicker (safer)
    assert df_defect.h_joint_mm.iloc[-1] > df_defect.h_joint_mm.iloc[0]

    df_tp = sensitivity_throughput_value(m, s, st)
    assert not df_tp.empty
    # As press time gets more expensive, optimal cut shifts thicker
    assert df_tp.h_joint_mm.iloc[-1] >= df_tp.h_joint_mm.iloc[0]

    df_tariff = sensitivity_electricity_tariff(m, s, st)
    assert not df_tariff.empty


def test_report_generation(st: JointSettings):
    report_text = generate_report(st)
    assert "Task 3.4: Joint Objective with Dead-Cycle Time and Pump Energy" in report_text
    assert "Done-When Sensitivity Analysis" in report_text
    assert "Single-objective cut" in report_text
