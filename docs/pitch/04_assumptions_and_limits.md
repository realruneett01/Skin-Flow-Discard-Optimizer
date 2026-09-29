# Press Value Platform — Assumptions and Limits

> This document lists every material assumption made by the platform and the
> conditions under which results may not transfer to a real press. It is a
> required companion to any use of numbers from `03_results.md`.

---

## Section 1 — Digital Twin Assumptions

### 1.1 Force-Curve Shape

**Assumption:** Skin-flow onset is indicated by an upturn (positive gradient
change) in the ram pressure signal. This is consistent with published tribology
literature for aluminium on steel billets.

**Limit:** Some presses, die geometries, or alloy combinations may show a
pressure-drop signature at onset rather than an upturn. The GLR onset detector
supports both directions via its `direction` parameter (see
`core/observer/onset.py`). A pilot must determine the correct direction from
real force curves.

**Config file:** `config/press.yaml` → `onset_direction`

---

### 1.2 Billet and Tooling Parameters

**Assumption:** Default billet diameter 178 mm, length 600 mm, mass 95 kg (AA6063/6082
density). Extrusion ratio 30–60:1. Liner temperature 400–460 °C.

**Limit:** Presses with significantly different billet geometry, extrusion ratios
outside this range, or ceramic liners will require parameter re-calibration.
All physical constants are in `config/press.yaml` and `config/coupling.yaml`.

---

### 1.3 Defect Model

**Assumption:** Skin-flow contamination probability follows a sigmoid function of
`(h_actual − h_crit)`. The `h_crit` parameter (critical onset depth) is drawn
from a hierarchical Bayesian prior fit on simulated data.

**Limit:** The sigmoid parameters and `h_crit` distribution are from the digital
twin, not from matched defect labels on a real press. The defect rate numbers in
`03_results.md` (0.56/1000) are simulation estimates and may differ substantially
on real equipment until calibrated against actual quality-gate records.

**Source:** `sim/defect_model.py`, `core/estimator/hierarchical.py`

---

### 1.4 Alloy Coverage

**Assumption:** The hierarchical Bayesian model is calibrated on AA6063 and AA6082
only. This covers a large fraction of commercial extruded profile production.

**Limit:** AA6061, AA6101, AA7075, AA2024, and other alloys are outside the
calibrated library. Requests for these alloys trigger a CRITICAL_FALLBACK (see
`core/safety.py`, `SafetyGate.validate_process_context`). A calibration run of
200–500 billets per alloy is required before advisory operation.

---

## Section 2 — Statistical and Modelling Assumptions

### 2.1 Conformal Prediction Coverage

**Assumption:** The adaptive conformal prediction wrapper provides 90% coverage
of `h_crit` on any billet drawn from the same distribution as the calibration
set (exchangeability assumption).

**Limit:** Coverage is empirically valid on the rolling-origin test folds
(91.4%, 95% CI [90.9%, 92.1%]). If real press conditions shift outside the
simulator's scenario envelope, coverage may drop until the conformal calibration
set is refreshed with real-press data.

**Source:** `core/estimator/pipeline.py` → `CutEngine`, `eval/backtest.py`

---

### 2.2 Multivariate Monitor Sensitivity

**Assumption:** The Hotelling T², SPE (Q-statistic), and MEWMA control charts
are calibrated on the Phase 2 feature distribution from the healthy-baseline
scenario. Alarm thresholds are set at α = 0.01 per chart.

**Limit:** Two scenarios showed late detection (liner_scale, sensor_gain_drift)
because their fault modes cause rapid step-changes rather than gradual drift.
The MEWMA's exponential weighting λ and SPE threshold may need per-site tuning.

**Source:** `aware/monitor.py`, `config/press.yaml` → `mewma_lambda`

---

### 2.3 Bayesian Fault Classifier

**Assumption:** The fault-class library covers eight named fault signatures
(die_wear, liner_scale, lubricant_loss, temperature_drift, encoder_offset,
flash_spike, supply_pressure_sag, sensor_gain_drift) plus an out-of-library
(OOL) reject class. All signatures are from the digital twin.

**Limit:** Real press faults may have different feature signatures, multi-fault
combinations, or classes not in the library. All OOL cases trigger the NOVEL_FAULT
safety fallback (see `artifacts/stress_summary.json`).

**Source:** `aware/fault_id.py`, `artifacts/fault_signatures.npz`

---

## Section 3 — Economic Assumptions

### 3.1 Metal Spread

**Assumption:** Default metal price 2.60 EUR/kg (primary aluminium gate price),
remelt credit 2.10 EUR/kg → spread 0.50 EUR/kg.

**Limit:** LME aluminium spot prices fluctuate. At spread ≤ 0 EUR/kg the SKDO
module produces zero economic return. The ROI calculator in `roi/dashboard.py`
allows real-time adjustment; the spread sensitivity table in `03_results.md`
shows values at spreads 0.20–0.80 EUR/kg.

**Source:** `config/economics.yaml` → `billet_price_per_kg`, `remelt_credit_per_kg`

---

### 3.2 DCTO and HPEO Baselines

**Assumption:** Dead-cycle saving baseline: 1.2–2.5 s/cycle. Pump energy saving
baseline: 0.8–1.8 kWh/cycle. Both derived from literature on hydraulic extrusion
press efficiency studies, not from measured press data.

**Limit:** DCTO and HPEO savings have not been validated by a simulation backtest
equivalent to the SKDO rolling-origin protocol. Their ROI ranges are wider
(the Low/Expected/High intervals reflect literature uncertainty, not empirical
validation uncertainty). These modules require independent pilot validation.

**Source:** `roi/engine.py` → `BASELINE_DCTO_*`, `BASELINE_HPEO_*`

---

### 3.3 Annual Production Volume

**Assumption:** 309,600 cycles/year = 43 cycles/hr × 7,200 operating hours/year.

**Limit:** Press-specific utilisation rates and scheduled maintenance windows may
reduce effective operating hours. The ROI calculator accepts any cycles/year
value between 100,000 and 500,000.

**Source:** `config/economics.yaml` → `cycles_per_hour`, `operating_hours_per_year`

---

## Section 4 — Operational Limits

### 4.1 Advisory-Only Constraint

The platform **never writes to any press actuator, setpoint, or control register.**
It publishes `Predictor.*` tags on the OPC-UA namespace. Bypassing this
constraint in any integration would require a separate safety analysis under
IEC 61511 (functional safety for process industry SIS) or equivalent.

### 4.2 Hard Output Bounds

`Predictor.ButtCutMm` is always clamped to [12.0, 60.0] mm by the SafetyGate.
If any validation check fails, the output falls back to 40.0 mm (the static
baseline). There is no condition under which the platform recommends a cut
outside these bounds.

**Source:** `config/defect.yaml` → `min_cut_mm`, `max_cut_mm`, `static_cut_mm`

### 4.3 Online Learning

The platform does not perform online learning or update its own model weights
during operation. All model parameters are frozen after calibration.
Periodic re-calibration (suggested: every 6–12 months or after major tooling
changes) is needed to maintain accuracy.

### 4.4 Streaming Latency

The p99 latency budget is 200 ms per billet decision. This is verified under
simulation load (measured 133 ms p99 in streaming mock runs). Real OPC-UA
network jitter and host load may increase latency; a dedicated monitoring task
should track `Predictor.*` tag update intervals.

---

## Summary Checklist for Pilot Scoping

| Item | What is needed | Config location |
|---|---|---|
| Force-curve direction | 50–100 billets with matched onset labels | `config/press.yaml → onset_direction` |
| Alloy calibration | 200–500 billets per new alloy | `core/estimator/hierarchical.py` |
| Defect labelling | 3–6 months matched quality-gate records | `sim/defect_model.py` |
| OPC-UA tag mapping | Signal contract review with plant DCS team | `contracts/signals.yaml` |
| Latency measurement | 1-week production monitoring run | `service/run.py → ServiceHealth` |
| Monitor re-calibration | After major tooling change or alloy shift | `aware/monitor.py → MultivariateMonitor.fit()` |
| MEWMA / threshold tuning | Site-specific false-alarm rate target | `config/press.yaml → mewma_lambda` |
