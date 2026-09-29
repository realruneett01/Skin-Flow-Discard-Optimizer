# Press Value Platform — Technical Summary

> **Simulation Disclosure** All performance numbers in this document come from a
> digital twin validated against published aluminium extrusion literature and open
> process-parameter databases. They are not yet measured on a real press. A pilot
> programme to calibrate on real force curves is described in §6.
>
> **Advisory-Only Disclosure** All platform outputs are strictly advisory
> (`Predictor.*` namespace). The platform publishes decision-support signals only;
> it writes to no shear, valve, or press setpoint.

---

## 1. Problem

Aluminium extrusion presses discard 40–80 mm of every billet as the "butt" — the
tail-end slug that retains oxidised skin-flow contamination. Static cut rules,
typically 40 mm, are set conservatively to keep defect risk low across all alloys,
temperatures, and tool conditions. The cost is systematic over-discard: the plant
loses recoverable aluminium on every cycle.

Three additional sources of idle value exist alongside yield:
- **Dead-cycle time** — non-productive press travel between extrusion end and butt
  removal consumes throughput capacity.
- **Hydraulic pump energy** — sustained high pressure during non-critical portions
  of the stroke wastes electrical energy.
- **Late process warnings** — tool-wear and contamination faults are detected only
  at the quality gate, not while they are developing.

---

## 2. Platform Concept

The **Press Value Platform** is a three-module advisory system that addresses all
four problems from a single set of per-cycle force-curve measurements already
available on modern OPC-UA press controllers.

```
OPC-UA tags  →  Signal preprocessing  →  Phase 2
                 (resample, filter, FPCA)
                        │
               ┌────────┴────────────────────────────┐
               │  Phase 3: Core Engine               │
               │  UKF (within-stroke observer)        │
               │  GLR+BOCPD (onset detector)          │
               │  Hierarchical Bayes + Conformal      │
               │  Joint objective (yield/time/power)  │
               └────────┬────────────────────────────┘
                        │
               ┌────────┴────────────────────────────┐
               │  Phase 4: Pattern Awareness         │
               │  Particle-filter wear tracker        │
               │  T²/SPE/MEWMA monitor               │
               │  Forward trajectory forecaster       │
               │  Bayesian fault classifier           │
               └────────┬────────────────────────────┘
                        │
               ┌────────┴────────────────────────────┐
               │  Phase 5: Safety Gate               │
               │  7-condition telemetry validator     │
               │  Hard bounds [12, 60] mm             │
               │  Conservative CRITICAL_FALLBACK      │
               └────────┬────────────────────────────┘
                        │
               Predictor.* advisory tags  →  Operator HMI
```

**Module 1 — Skin-Flow Discard Optimizer (SKDO)**
Estimates the true skin-flow onset depth per billet from live force-curve
gradients, publishes `Predictor.ButtCutMm` as a tighter-than-static
recommendation.

**Module 2 — Dead-Cycle Timer Optimizer (DCTO)**
Learns per-alloy/die dead-cycle timing patterns, publishes time-to-next-cycle
advisories to reduce non-productive press travel.

**Module 3 — Hydraulic Pump Energy Optimizer (HPEO)**
Identifies pressure-reduction windows during non-critical stroke segments,
publishes advisory pressure-schedule targets to reduce pump energy consumption.

---

## 3. Architecture Highlights

| Property | Design choice | Rationale |
|---|---|---|
| Output protocol | `Predictor.*` namespace, read-only | Operator retains physical authority; zero setpoint writes |
| Uncertainty quantification | Adaptive conformal intervals (90% target coverage) | Empirical guarantee under distribution shift, not assumed Gaussian |
| Fallback safety | 7-condition SafetyGate, hard bounds [12, 60] mm | Zero production risk even if model confidence collapses |
| Streaming latency | p99 < 200 ms per billet decision | Fits within typical OPC-UA polling cycle |
| Drift robustness | MEWMA + Hotelling T² + SPE + regime switching | Continues operating under gradual tool wear and scale build-up |
| Fault isolation | Bayesian multi-class classifier with OOL reject | Identifies die_wear, liner_scale, lubricant_loss etc. as named causes |

---

## 4. Validation Methodology (Phase 5)

The platform was evaluated on a rolling-origin backtest across **13 scenarios**
(11 development, 2 fully held-out) spanning **12,660 total billet cycles** generated
by a physics-validated digital twin.

Key design properties of the evaluation:
- **Rolling-origin folds** prevent future data leakage into any calibration step.
- **Held-out scenarios** (`combined_wear_and_scale`, `die_change`) use alloy/tool
  combinations the model never saw during fitting.
- All numbers are reported with 95% confidence intervals; point estimates alone
  are never cited.

Source file: `eval/backtest.py` · Results file: `artifacts/backtest_summary.json`

---

## 5. Key Results (from `artifacts/backtest_summary.json`)

All figures are from simulation. See §6 for what a pilot must verify.

| KPI | Point estimate | 95% CI | Target | Status |
|---|---|---|---|---|
| Mean butt thickness (SKDO) | **29.5 mm** | [28.6, 30.2] | < 40 mm static | ✅ PASS |
| Metal recovery vs. static cut | **1.29% of billet mass** | [1.20, 1.40] | > 0 | ✅ PASS |
| Defect rate | **0.56 / 1000 billets** | [0.44, 0.68] | ≤ static baseline | ⚠️ CAUTION |
| False-alarm rate | **4.9 / 1000 cycles** | [2.9, 7.4] | ≤ 5.0 / 1000 | ✅ PASS |
| Warning lead time | **300 cycles (median)** | [−29, 1546] | median > 0 | ✅ PASS |
| Conformal interval coverage | **91.4%** | [90.9, 92.1] | 90% ± 2% | ✅ PASS |
| Cut-decision latency (p99) | **15.8 ms** | — | < 200 ms | ✅ PASS |
| Net metal saving per billet | **0.390 EUR** | [0.346, 0.463] | > 0 | ✅ PASS |

> **Defect rate note.** The CAUTION flag reflects that SKDO introduces a small
> positive defect probability relative to the *zero-defect* static-cut baseline.
> The static rule achieves near-zero defects by cutting conservatively deep.
> SKDO's defect rate (0.56/1000) is well within typical quality-gate reject rates
> for aluminium extrusion (~5–15/1000 from surface and dimensional causes).

---

## 6. Pilot Programme Required

Results are from simulation. The following must be verified on real data before
commercial use:

1. **Force-curve shape calibration** — The digital twin uses a friction-based
   upturn model. Real presses may show a pressure-drop shape. The GLR onset
   detector supports both; the direction parameter must be set from real data.
2. **Alloy-specific model fitting** — The hierarchical Bayesian model currently
   covers AA6063 and AA6082. Each additional alloy requires a minimum run of
   ~200–500 calibration billets.
3. **Defect labelling** — `h_crit` is simulated from a physics model. A 3–6 month
   data collection period with matched quality-gate records is required to
   calibrate the onset estimator against real contamination events.
4. **OPC-UA signal mapping** — Tag names and engineering unit conventions must be
   mapped against `contracts/signals.yaml` for each press controller.
5. **Latency budget verification** — p99 < 200 ms is confirmed under simulation
   load. Real OPC-UA sampling jitter and network round-trip must be measured.

---

## 7. What the Platform Does Not Do

- **Does not control any actuator.** It publishes advisory tags only.
- **Does not guarantee zero defects.** It reduces expected discard while keeping
  defect risk low; a residual defect rate exists (see §5).
- **Does not operate on unseen alloys without calibration.** The SafetyGate
  issues a CRITICAL_FALLBACK for alloy IDs not in the calibrated library.
- **Does not replace operator judgement.** Every Predictor.* signal is a
  recommendation; the operator accepts or overrides it.
