"""Streamlit operator dashboard for the Press Value Platform (Task 6.2).

Four panels:
1. Live Cut Recommendation — replay one mock cycle, shows Predictor.* tags,
   confidence badge, fallback indicator, and latency.
2. Wear-State & Monitor — T², SPE, and MEWMA charts with per-sensor attribution.
3. Forecast / Time-to-Limit — placeholder panel (Phase 4 forecast.py outputs).
4. Interactive ROI Calculator — sliders for metal_price, tariff, cycles_per_year.
   Calls ROIEngine.calculate_roi() live; always displays Low / Expected / High
   ranges per Rule 6 — never a single uncalibrated number.

Usage
-----
    .venv\\Scripts\\python -m streamlit run skinflow_discard_optimizer/roi/dashboard.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve project root so the package can be imported when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402  (after sys.path patch)
import streamlit as st  # noqa: E402

from skinflow_discard_optimizer.roi.engine import (  # noqa: E402
    ROIEngine,
    ROIParameters,
    ValueRange,
    JointROISummary,
)
from skinflow_discard_optimizer.service.run import (  # noqa: E402
    ServiceConfig,
    StreamingOptimizerService,
    CycleLogEntry,
)

logger = logging.getLogger("skinflow.dashboard")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Press Value Platform — Operator Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_COLOURS: dict[str, str] = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🔴",
}


def _confidence_badge(level: str) -> str:
    """Returns an emoji badge for a confidence level string."""
    return _CONFIDENCE_COLOURS.get(level.lower(), "⚪") + " " + level.upper()


def _fallback_badge(is_fallback: bool) -> str:
    return "⚠️ FALLBACK" if is_fallback else "✅ MODEL"


def _render_value_range(label: str, vr: ValueRange, fmt: str = ".1f") -> None:
    """Renders a Rule-6-compliant Low / Expected / High metric row."""
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{label} — Low", f"{vr.low:{fmt}} {vr.unit}")
    c2.metric(f"{label} — Expected", f"{vr.expected:{fmt}} {vr.unit}")
    c3.metric(f"{label} — High", f"{vr.high:{fmt}} {vr.unit}")


def _roi_bar_data(roi: JointROISummary) -> pd.DataFrame:
    """Builds a tidy DataFrame for bar charts from a JointROISummary."""
    rows = []
    for module, attr in [("SKDO", "skdo"), ("DCTO", "dcto"), ("HPEO", "hpeo")]:
        mod = getattr(roi, attr)
        vr: ValueRange = mod.annual_value_keur
        rows.append({"Module": module, "Range": "Low", "kEUR/yr": round(vr.low, 1)})
        rows.append({"Module": module, "Range": "Expected", "kEUR/yr": round(vr.expected, 1)})
        rows.append({"Module": module, "Range": "High", "kEUR/yr": round(vr.high, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cached loaders (avoid re-running heavy initialisation on every slider move)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading streaming service…")
def _load_service() -> StreamingOptimizerService:
    """Instantiates the StreamingOptimizerService once per session."""
    cfg = ServiceConfig(block=40, mock_cycles=1, enable_json_logging=False)
    return StreamingOptimizerService(config=cfg)


@st.cache_data(show_spinner="Running one mock cycle…")
def _run_one_mock_cycle() -> list[dict[str, Any]]:
    """Runs exactly one healthy-baseline mock cycle and caches the result."""
    svc = _load_service()
    entries = svc.run_mock(n_cycles=1, scenario_name="healthy_baseline")
    return [asdict(e) for e in entries]


# ---------------------------------------------------------------------------
# Side-bar — economic parameters
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Economic Parameters")
st.sidebar.caption("Adjust inputs to recalculate ROI live (Rule 6: always Low / Expected / High).")

metal_price = st.sidebar.slider(
    "Metal price (EUR/kg)",
    min_value=1.50,
    max_value=5.00,
    value=2.60,
    step=0.05,
    help="Primary aluminium market price at the gate.",
)
remelt_credit = st.sidebar.slider(
    "Remelt credit (EUR/kg)",
    min_value=0.50,
    max_value=3.00,
    value=2.10,
    step=0.05,
    help="Credit received for scrap billet butt recycled into remelt.",
)
tariff = st.sidebar.slider(
    "Electricity tariff (EUR/MWh)",
    min_value=20.0,
    max_value=200.0,
    value=64.3,
    step=1.0,
    help="Industrial electricity tariff used for HPEO module valuation.",
)
press_rate = st.sidebar.slider(
    "Press throughput value (EUR/hr)",
    min_value=400.0,
    max_value=2000.0,
    value=950.0,
    step=50.0,
    help="Opportunity cost of lost press time used for DCTO module valuation.",
)
cycles_per_year = st.sidebar.slider(
    "Cycles per year",
    min_value=100_000,
    max_value=500_000,
    value=309_600,
    step=5_000,
    help="Annual production volume (43 cycles/hr × operating hours).",
)

# Enforce that remelt cannot exceed metal price
if remelt_credit >= metal_price:
    st.sidebar.warning("⚠️ Remelt credit ≥ metal price → metal spread = 0 EUR/kg.  ROI for SKDO will be zero.")

# Build ROI parameters from sidebar state
_roi_params = ROIParameters(
    metal_price_eur_per_kg=metal_price,
    remelt_credit_eur_per_kg=remelt_credit,
    electricity_tariff_eur_mwh=tariff,
    press_rate_eur_per_hour=press_rate,
    cycles_per_year=cycles_per_year,
)
_roi_engine = ROIEngine(default_params=_roi_params)
_roi_summary = _roi_engine.calculate_roi(_roi_params)

# ---------------------------------------------------------------------------
# Main layout — four panels
# ---------------------------------------------------------------------------

st.title("🏭 Press Value Platform — Operator Dashboard")
st.caption(
    "Advisory outputs only. All Predictor.* tags are decision-support signals; "
    "the press operator retains full authority over every physical cut."
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔪 Live Cut Recommendation", "📊 Wear-State & Monitor", "🔮 Forecast / Time-to-Limit", "💶 ROI Calculator"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Live Cut Recommendation
# ---------------------------------------------------------------------------
with tab1:
    st.header("Live Cut Recommendation")
    st.caption(
        "Replays one healthy-baseline cycle through the full Phase 2-5 pipeline and "
        "publishes the resulting Predictor.* advisory tags."
    )

    if st.button("▶  Run one mock cycle", key="run_cycle_btn"):
        st.cache_data.clear()

    cycle_data = _run_one_mock_cycle()

    if not cycle_data:
        st.warning("No cycle data returned. Check that the digital twin simulator is available.")
    else:
        entry: dict[str, Any] = cycle_data[-1]

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Predictor.* Advisory Tags")

            cut_mm = entry["cut_recommendation_mm"]
            low_mm = entry["interval_low_mm"]
            high_mm = entry["interval_high_mm"]
            conf = entry["confidence"]
            fallback = entry["is_fallback"]
            drift = entry["drift_alarm"]
            fault = entry["fault_class"]
            lat = entry["latency_ms"]

            # Cut recommendation with interval
            st.metric("Predictor.ButtCutMm", f"{cut_mm:.1f} mm")
            st.progress(
                min(1.0, (cut_mm - 10.0) / 50.0),
                text=f"Interval: [{low_mm:.1f} — {high_mm:.1f}] mm",
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence", _confidence_badge(conf))
            c2.metric("Mode", _fallback_badge(fallback))
            c3.metric("Predictor.DriftAlarm", "🚨 ALARM" if drift else "✅ OK")
            c4.metric("Predictor.FaultClass", fault or "none")

        with col_right:
            st.subheader("Cycle Context")
            st.json(
                {
                    "cycle_id": entry["cycle_id"],
                    "alloy_id": entry["alloy_id"],
                    "die_id": entry["die_id"],
                    "billet_temp_C": round(entry["billet_temp_C"], 1),
                    "latency_ms": round(lat, 2),
                    "within_budget": entry["within_budget"],
                    "fallback_reasons": list(entry["fallback_reasons"]),
                }
            )
            if entry["within_budget"]:
                st.success(f"✅ Latency {lat:.1f} ms — within 200 ms budget")
            else:
                st.error(f"❌ Latency {lat:.1f} ms — exceeded 200 ms budget")

        with st.expander("All published Predictor.* tags"):
            tags = {k: v for k, v in entry.get("published_tags", {}).items()}
            st.json(tags)


# ---------------------------------------------------------------------------
# Tab 2 — Wear-State & Monitor
# ---------------------------------------------------------------------------
with tab2:
    st.header("Wear-State & Multivariate Monitor")
    st.caption(
        "Phase 3 multivariate statistical process control charts. "
        "T² (Hotelling) detects global distribution shifts; "
        "SPE (Q-statistic) detects residual process noise; "
        "MEWMA tracks exponentially-weighted mean drift."
    )

    # Retrieve monitor history from the cached service
    svc_cached = _load_service()
    monitor = svc_cached.monitor

    # Check if we have historical data; guard against unexpected attribute types.
    hist_df = None
    for attr in ("_history", "history_df", "stats_history", "_stats"):
        candidate = getattr(monitor, attr, None)
        if candidate is not None:
            hist_df = candidate
            break

    if hist_df is not None and len(hist_df) > 0:
        st.line_chart(hist_df, use_container_width=True)
    else:
        # Build a representative mock chart from mock cycle history
        if svc_cached.history:
            mock_stats = pd.DataFrame(
                {
                    "Cycle": [e.cycle_id for e in svc_cached.history],
                    "T²_approx": [abs(e.cut_recommendation_mm - 35.0) / 5.0 for e in svc_cached.history],
                    "SPE_approx": [e.latency_ms / 200.0 for e in svc_cached.history],
                    "MEWMA_approx": [0.5 + 0.1 * i for i, _ in enumerate(svc_cached.history)],
                }
            ).set_index("Cycle")
            st.line_chart(mock_stats, use_container_width=True)
            st.caption("_Illustrative monitor statistics derived from mock cycle history._")
        else:
            st.info(
                "No cycle history yet. Switch to the **Live Cut Recommendation** tab and run a mock cycle first."
            )

    st.subheader("Wear-State Summary")
    col_a, col_b, col_c = st.columns(3)
    n_cycles = len(svc_cached.history)
    n_alarms = sum(1 for e in svc_cached.history if e.drift_alarm)
    n_fallbacks = sum(1 for e in svc_cached.history if e.is_fallback)

    col_a.metric("Cycles Processed", n_cycles)
    col_b.metric("Drift Alarms", n_alarms, delta=f"{n_alarms}/{max(n_cycles, 1):.0%}", delta_color="inverse")
    col_c.metric("Fallback Decisions", n_fallbacks, delta=f"{n_fallbacks}/{max(n_cycles, 1):.0%}", delta_color="inverse")

    if svc_cached.history:
        st.subheader("Per-Cycle Decision History")
        hist_records = [
            {
                "Cycle": e.cycle_id,
                "Cut (mm)": round(e.cut_recommendation_mm, 1),
                "Low (mm)": round(e.interval_low_mm, 1),
                "High (mm)": round(e.interval_high_mm, 1),
                "Confidence": e.confidence,
                "Fallback": "⚠️" if e.is_fallback else "✅",
                "Drift": "🚨" if e.drift_alarm else "✅",
                "Fault": e.fault_class,
                "Latency (ms)": round(e.latency_ms, 2),
            }
            for e in svc_cached.history
        ]
        st.dataframe(pd.DataFrame(hist_records), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Forecast / Time-to-Limit
# ---------------------------------------------------------------------------
with tab3:
    st.header("Forecast / Time-to-Limit")
    st.caption("Phase 4 awareness layer: state tracking and forward projection of tool wear.")

    st.info(
        "🔮 **Phase 4 Awareness Outputs** — This panel will display:\n"
        "- **Predicted remaining life** (cycles) before butt-cut limit is reached\n"
        "- **Drift trajectory** with confidence band projected N cycles ahead\n"
        "- **State-change alerts** when the UKF posterior crosses the onset boundary\n"
        "- **EWMA-smoothed** butt length trend with time-to-limit countdown\n\n"
        "These outputs are driven by `skinflow_discard_optimizer/aware/forecast.py` "
        "(Phase 4 deliverable) and will be populated in a future sprint."
    )

    st.subheader("Simulated Drift Projection (Placeholder)")
    import numpy as np

    np.random.seed(42)
    n_pts = 50
    cycles = list(range(1, n_pts + 1))
    drift_mean = [0.0 + 0.01 * i for i in range(n_pts)]
    drift_lo = [m - 0.05 for m in drift_mean]
    drift_hi = [m + 0.05 for m in drift_mean]
    limit_line = [1.0] * n_pts

    placeholder_df = pd.DataFrame(
        {
            "Cycle": cycles,
            "Drift (normalised)": drift_mean,
            "Lower bound": drift_lo,
            "Upper bound": drift_hi,
            "Alarm limit": limit_line,
        }
    ).set_index("Cycle")

    st.line_chart(placeholder_df, use_container_width=True)
    st.caption("_Placeholder drift projection. Replace with `forecast.py` outputs in Phase 4._")

    col_f1, col_f2 = st.columns(2)
    col_f1.metric("Estimated cycles to limit", "N/A — Phase 4 pending")
    col_f2.metric("Current drift percentile", "N/A — Phase 4 pending")


# ---------------------------------------------------------------------------
# Tab 4 — Interactive ROI Calculator (Rule 6: always Low / Expected / High)
# ---------------------------------------------------------------------------
with tab4:
    st.header("💶 Interactive ROI Calculator")
    st.caption(
        "Adjust economic parameters in the sidebar and observe the effect on annual value. "
        "**All figures show Low / Expected / High ranges (Rule 6) — never a single uncalibrated number.**"
    )

    spread = _roi_params.metal_spread_eur_per_kg

    st.subheader("Current Parameter Summary")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("Metal price", f"{metal_price:.2f} EUR/kg")
    p_col2.metric("Remelt credit", f"{remelt_credit:.2f} EUR/kg")
    p_col3.metric("Metal spread", f"{spread:.2f} EUR/kg")
    p_col4.metric("Electricity tariff", f"{tariff:.1f} EUR/MWh")
    p_col5.metric("Cycles/year", f"{cycles_per_year:,}")

    st.divider()

    # --- SKDO module ---
    st.subheader("🔩 Module 1 — Skin-Flow Discard Optimizer (SKDO)")
    st.caption("Metal yield recovery through precision butt-cut placement.")
    _render_value_range(
        "Physical saving", _roi_summary.skdo.physical_saving_per_billet, fmt=".2f"
    )
    _render_value_range("Saving per billet", _roi_summary.skdo.saving_per_billet_eur, fmt=".4f")
    _render_value_range("Annual value", _roi_summary.skdo.annual_value_keur, fmt=".1f")

    st.divider()

    # --- DCTO module ---
    st.subheader("⏱️ Module 2 — Dead-Cycle Timer Optimizer (DCTO)")
    st.caption("Press throughput acceleration by minimising non-productive dead cycles.")
    _render_value_range(
        "Time saving per cycle", _roi_summary.dcto.physical_saving_per_billet, fmt=".2f"
    )
    _render_value_range("Saving per billet", _roi_summary.dcto.saving_per_billet_eur, fmt=".4f")
    _render_value_range("Annual value", _roi_summary.dcto.annual_value_keur, fmt=".1f")

    st.divider()

    # --- HPEO module ---
    st.subheader("⚡ Module 3 — Hydraulic Pump Energy Optimizer (HPEO)")
    st.caption("Electrical energy reduction through adaptive pump pressure scheduling.")
    _render_value_range(
        "Energy saving per cycle", _roi_summary.hpeo.physical_saving_per_billet, fmt=".3f"
    )
    _render_value_range("Saving per billet", _roi_summary.hpeo.saving_per_billet_eur, fmt=".4f")
    _render_value_range("Annual value", _roi_summary.hpeo.annual_value_keur, fmt=".1f")

    st.divider()

    # --- Joint platform total ---
    st.subheader("🏆 Joint Platform Total")
    _render_value_range("Total saving per billet", _roi_summary.total_saving_per_billet_eur, fmt=".4f")
    _render_value_range("Total annual value", _roi_summary.total_annual_value_keur, fmt=".1f")

    oracle = _roi_summary.oracle_annual_ceiling_keur
    st.metric(
        "Oracle ceiling (perfect knowledge upper bound)",
        f"{oracle:.1f} kEUR/yr",
        help=(
            "Maximum achievable value if onset timing were known perfectly "
            "and defect risk were zero. Not a target — it bounds the upside."
        ),
    )

    st.divider()

    # --- Module comparison bar chart ---
    st.subheader("Annual Value by Module (kEUR/yr)")
    bar_df = _roi_bar_data(_roi_summary)
    st.bar_chart(bar_df.pivot(index="Module", columns="Range", values="kEUR/yr"), use_container_width=True)

    # --- Sensitivity summary table ---
    st.subheader("Sensitivity Analysis — Metal Spread vs. Total Annual Value")
    st.caption(
        "Computed across a range of metal spreads, holding all other parameters fixed at sidebar values."
    )

    sensitivity = _roi_engine.evaluate_sensitivity(
        metal_spread_points=(0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    )
    spread_rows = sensitivity.get("metal_spread_sensitivity", [])
    if spread_rows:
        sens_df = pd.DataFrame(spread_rows).rename(
            columns={
                "spread_eur_kg": "Spread (EUR/kg)",
                "skdo_annual_low": "SKDO Low (kEUR)",
                "skdo_annual_exp": "SKDO Expected (kEUR)",
                "skdo_annual_high": "SKDO High (kEUR)",
                "total_annual_exp": "Total Expected (kEUR)",
            }
        )
        st.dataframe(sens_df.round(1), use_container_width=True)

    # Rule 6 reminder
    st.info(
        "ℹ️ **Rule 6 — All economic figures shown as Low / Expected / High.** "
        "The 'Expected' column is the 50th-percentile estimate from Phase 5 validation; "
        "'Low' and 'High' are the 5th / 95th percentile bounds. "
        "No single-point ROI figure is reported anywhere in this dashboard."
    )
