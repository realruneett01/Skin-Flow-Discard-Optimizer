# Press Value Platform — 90-Second Demo Script

> **Presenter guidance.** This script walks through five scenes in ≤ 90 seconds
> using the Streamlit dashboard and a mock simulation run. All data shown is from
> the digital twin (`sim/`). Remind the audience of this at scenes 1 and 5.

---

## Pre-Demo Setup (run before the audience arrives)

```powershell
# 1. Start the dashboard
cd C:\Users\realr\OneDrive\Desktop\SKDO
.venv\Scripts\python -m streamlit run skinflow_discard_optimizer/roi/dashboard.py

# 2. Browser opens at http://localhost:8501 — navigate to Tab 1
```

Default sidebar values:
- Metal price: 2.60 EUR/kg
- Remelt credit: 2.10 EUR/kg
- Electricity tariff: 64.3 EUR/MWh
- Press rate: 950 EUR/hr
- Cycles per year: 309,600

---

## Scene 1 — Healthy Baseline Run (0:00 – 0:15)

**Narrative:** *"Here is a healthy press cycle — the platform processes one billet
in real time and publishes its advisory recommendation."*

**Action:** On Tab 1 ("🔪 Live Cut Recommendation"), click **▶ Run one mock cycle**.

**What to show:**
- `Predictor.ButtCutMm` → **~28.4 mm** (vs. 40 mm static rule — a ~11 mm saving)
- Confidence badge → 🟢 **HIGH**
- Mode badge → ✅ **MODEL** (not fallback)
- Latency → **< 200 ms** (green status)
- Interval bar → e.g. [26.9 – 29.9] mm — this is the conformal interval

**Say:** *"The model recommends cutting at 28 mm instead of 40 mm. It is a
suggestion — the operator decides. The interval shows the 90% confidence bound
on where the skin-flow boundary actually is. Results from simulation."*

---

## Scene 2 — Drift Developing (0:15 – 0:35)

**Narrative:** *"Now watch what happens as tool wear develops over hundreds of
cycles. The monitor catches it before a defect occurs."*

**Action:** Switch to Tab 2 ("📊 Wear-State & Monitor"). Point to the T²/MEWMA
chart.

**What to show (mock history if available, or illustrative chart):**
- T² and MEWMA trends rising over cycles
- `Drift Alarms` counter (> 0 after mock runs with drift scenarios)
- Decision history table — note confidence level changing from HIGH → MEDIUM

**Say:** *"Three statistics track the process continuously: Hotelling T² for
global distribution shifts, SPE for residual noise, and MEWMA for slow mean
drift. In the die-wear scenario, MEWMA raised an alarm 222 cycles — about
5 hours — before the quality limit was breached."*

> **Data source:** `backtest_summary.json` → `lead_time_metrics` → `die_wear`
> (first_alarm_cycle 4000, limit_breach_cycle 4222, lead_time 222 cycles)

---

## Scene 3 — Early Warning Issued (0:35 – 0:50)

**Narrative:** *"The platform does not just detect drift — it identifies the cause
and gives the operator a named fault to investigate."*

**Action:** Remain on Tab 2. Point to the `Fault` column in the decision table
or the `Predictor.FaultClass` metric.

**What to show:**
- `Predictor.FaultClass` → e.g. `die_wear_gradual` or `temperature_drift`
- `Predictor.DriftAlarm` → 🚨 ALARM

**Say:** *"The Bayesian fault classifier maps the multivariate pattern to one of
eight named failure modes — die wear, liner scale, lubricant loss, temperature
drift, and so on. An out-of-library fault gets a NOVEL_FAULT flag and triggers
the conservative 40 mm fallback automatically."*

> **Data source:** `artifacts/stress_summary.json` → `heavy_drift` and
> `novel_fault` scenarios; `aware/fault_id.py`

---

## Scene 4 — Money Impact (0:50 – 1:15)

**Narrative:** *"Now let me show what this is worth at this plant's economics.
And I'll change the metal price to show you it updates live."*

**Action:** Switch to Tab 4 ("💶 ROI Calculator").

**What to show (at default slider values):**
- SKDO row: Low **107 kEUR/yr** | Expected **121 kEUR/yr** | High **143 kEUR/yr**
- Joint Total: Low **221 kEUR/yr** | Expected **294 kEUR/yr** | High **384 kEUR/yr**

**Action:** Move the `Metal price` slider from 2.60 to 3.20 EUR/kg.

**What to show:** SKDO and Joint totals update immediately. Expected SKDO jumps to
~193 kEUR/yr. Show the sensitivity table scrolling.

**Say:** *"Every figure shown is a Low/Expected/High range — not a single
promised number. The Expected is the median from our Phase 5 evaluation. As
the metal price changes, the ROI updates live. We never report one number alone —
we show the full uncertainty range."*

> **Data source:** `artifacts/roi_summary.json`, `roi/engine.py`,
> `artifacts/backtest_summary.json` → `supporting_kpis[1]`

---

## Scene 5 — Safety and Limits (1:15 – 1:30)

**Narrative:** *"One last thing — this is designed to be safe by construction."*

**Stay on Tab 4 or switch back to Tab 1.**

**Say:** *"The output is always advisory — `Predictor.*` tags only, zero setpoint
writes. If telemetry is bad, the alloy is unknown, or the model confidence is
low, the SafetyGate clamps to the conservative 40 mm static rule automatically.
We tested this in 7 adversarial stress scenarios — missing data, frozen sensors,
timestamp errors, out-of-range pressure, unseen alloy, novel fault, heavy drift —
and in all 7 cases the fallback triggered correctly."*

*"All numbers I've shown are from a digital twin validated against published
aluminium extrusion literature. A pilot programme calibrating on real force
curves is required before production use — and we have a clear roadmap for that."*

> **Data source:** `artifacts/stress_summary.json`, `eval/stress.py`,
> `core/safety.py`; pilot requirements: `docs/pitch/04_assumptions_and_limits.md`

---

## Anticipated Questions

| Question | Answer pointer |
|---|---|
| "What happens if your model is wrong?" | SafetyGate → 40 mm fallback. Advisory only. See `core/safety.py`. |
| "What does the defect CAUTION flag mean?" | 0.56/1000 vs. industry 5–15/1000 total reject rate. See `03_results.md §KPI 3`. |
| "How long to calibrate on our press?" | 3–6 months for SKDO defect labelling; 200–500 billets/alloy. See `04_assumptions_and_limits.md §pilot`. |
| "Does it work on AA7075?" | Not yet — CRITICAL_FALLBACK for uncalibrated alloys. Can be added after ~500 billets. |
| "Can it control the shear automatically?" | No. Advisory only by design. Would require IEC 61511 SIL assessment. |
| "What's the hardware requirement?" | Any OPC-UA-capable server running Python 3.11+. p99 < 200 ms on a single CPU core. |
| "Is the ROI guaranteed?" | No. It is a range from simulation. Real values depend on metal spread, press utilisation, and pilot calibration quality. |
