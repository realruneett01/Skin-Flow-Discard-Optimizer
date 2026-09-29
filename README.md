# Skin-Flow Discard Optimizer (SKDO)

> **Third module of the Press Value Platform**
> Real-time skin-flow onset detection and per-billet optimal discard-cut recommendation engine for aluminium extrusion presses.

---

> [!IMPORTANT]
> **Advisory-only system.** All outputs are published exclusively to `Predictor.*` OPC-UA tags. This package **never writes** to any shear, valve, or press setpoint. The operator accepts or overrides every recommendation.

> [!NOTE]
> **Simulation disclosure.** All performance numbers in this document are derived from a physics-validated digital twin. No measurement from a real aluminium extrusion press has been used. See [§ Pilot Programme](#-pilot-programme-required) for what must be verified before production use.

---

## Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Platform Context](#-press-value-platform-context)
3. [Architecture](#-architecture)
4. [Core Algorithms](#-core-algorithms)
5. [Repository Layout](#-repository-layout)
6. [Signal Contract](#-signal-contract)
7. [Configuration](#-configuration)
8. [Evaluation Results](#-evaluation-results-phase-5)
9. [Safety Gate](#-safety-gate)
10. [KPIs](#-kpis)
11. [Integration Hazards](#-integration-hazards)
12. [Pilot Programme](#-pilot-programme-required)
13. [Installation & Running](#-installation--running)
14. [Testing](#-testing)
15. [Key Assumptions](#-key-assumptions)
16. [Limitations](#-limitations)

---

## ⚙️ Problem Statement

Aluminium extrusion presses discard 40–80 mm of every billet as the "butt" — the tail-end slug that retains oxidised skin-flow contamination. Static cut rules (typically 40 mm) are set conservatively to protect quality across all alloys, temperatures, and tool conditions. The cost is systematic over-discard: the plant loses recoverable aluminium on every cycle.

**What this module does:**
- Estimates the true skin-flow onset depth *per billet* from live force-curve gradients sampled at 1 kHz
- Publishes `Predictor.ButtCutMm` — a tighter-than-static cut recommendation with a calibrated uncertainty interval
- Raises drift and fault alarms *before* a quality-limit breach, giving operators intervention lead time

---

## 🏭 Press Value Platform Context

SKDO is the third module of the **Press Value Platform**, a three-module advisory system addressing the four main sources of idle value on modern aluminium extrusion presses:

| Module | Primary Lever | Expected Saving (Low / Expected / High) |
|---|---|---|
| **SKDO** — Skin-Flow Discard Optimizer | Metal yield via tighter discard cut | 107 / 121 / 143 kEUR/yr |
| **DCTO** — Dead-Cycle Timer Optimizer | Throughput via dead-cycle compression | 98 / 147 / 204 kEUR/yr |
| **HPEO** — Hydraulic Pump Energy Optimizer | Energy via pressure-reduction windows | 16 / 26 / 36 kEUR/yr |
| **Joint Total** | All three modules combined | **221 / 294 / 384 kEUR/yr** |

_Economics: aluminium spread 0.50 EUR/kg, tariff 64.3 EUR/MWh, press rate 950 EUR/hr, 309,600 cycles/yr._

The three modules share a common `contracts/signals.yaml` data contract and a shared ROI aggregation layer (`roi/engine.py`).

---

## 🏗️ Architecture

```
OPC-UA (1 kHz cap/rod pressure, ram position, temperatures)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2 — Signal Preprocessing                     │
│  Resample · Savitzky-Golay filter (adaptive window) │
│  FPCA basis · Feature extraction · Cycle store      │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────────────────────┐
        │  Phase 3 — Core Engine                  │
        │                                         │
        │  observer/ukf.py                        │
        │    Unscented Kalman Filter (UKF) over   │
        │    φ = [s, s·μ, F_tool] — within-stroke │
        │    state estimation of force-model      │
        │    parameters; θ recovered by unscented │
        │    transform                            │
        │                                         │
        │  observer/onset.py                      │
        │    GLR + BOCPD dual-test onset detector  │
        │    Detects skin-flow upturn in dF/dx    │
        │    and d²F/dx²; adaptive Lepski window  │
        │                                         │
        │  estimator/decision.py                  │
        │    Hierarchical Bayesian posterior over │
        │    h_crit (conjugate Gibbs sampler,     │
        │    partial pooling by die + alloy)      │
        │    → adaptive conformal prediction      │
        │    interval (90% empirical coverage)    │
        │                                         │
        │  estimator/pipeline.py                  │
        │    Joint cost minimisation:             │
        │    C(h) = c_m·ρ·A_b·h + c_d·P(h<h_crit)│
        │    outputs h* = Predictor.ButtCutMm    │
        └────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────────────────────┐
        │  Phase 4 — Pattern Awareness            │
        │  aware/state_tracker.py  Particle-filter│
        │    wear/tool-state tracker              │
        │  aware/monitor.py  T²/SPE/MEWMA SPC     │
        │  aware/forecast.py  Trajectory forecast │
        │  aware/fault_id.py  Bayesian multi-class│
        │    fault classifier (OOL-reject)        │
        └────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────────────────────┐
        │  Phase 5 — Safety Gate (core/safety.py) │
        │  7-condition validator · Hard bounds    │
        │  [12, 60] mm · CRITICAL_FALLBACK = 40 mm│
        └────────────┬────────────────────────────┘
                     │
        Predictor.ButtCutMm  ·  Predictor.ButtCutInterval
        Predictor.OnsetDepthMm  ·  Predictor.DriftAlarm
        Predictor.FaultClass  ·  Predictor.ConfidenceLevel
                     │
                Operator HMI  (read-only advisory)
```

---

## 🔬 Core Algorithms

### Unscented Kalman Filter (UKF) — `core/observer/ukf.py`

The UKF operates over the state vector **φ = [s, s·μ, F\_tool]** rather than the force-model parameter vector **θ = [σ\_scale, μ, F\_tool]** directly. Key rationale:

- The force model is bilinear in θ; the `s·μ` product is pinned far more tightly than `s` alone by the data, so the posterior in θ is a curved ridge that a Gaussian cannot represent. A UKF run directly on θ drifted ~20% from the exact posterior on the same data.
- In φ-space the measurement equation is **linear**, making the unscented update exact.
- θ and its covariance are recovered via an unscented transform after each update.

Ram speed is estimated as a least-squares slope over the trailing 1 s window (not a two-point finite difference), eliminating an ~1% errors-in-variables bias on σ\_scale.

### Onset Detector — `core/observer/onset.py`

Dual-test detector combining:
- **GLR (Generalised Likelihood Ratio)** test on dF/dx — detects the mean-shift in force-gradient when the upturn begins
- **BOCPD (Bayesian Online Change-Point Detection)** on d²F/dx² — confirms curvature change

Derivative estimation uses **Savitzky-Golay with Lepski-adaptive window selection** (κ = 3.5, windows 2.1–40 mm), chosen per position from local noise level. Error against noise-free truth: dF/dx ~2% mid-stroke, <1% in the gated tail; d²F/dx² ~5% in the tail.

Onset is defined as the thickness where F\_up reaches `upturn.amplitude_at_onset_N` (0.25 MN) — a defined physical criterion, not "first departure from noise".

### Hierarchical Bayesian Estimator — `core/estimator/hierarchical.py`

Partial pooling over die and alloy groups via a hand-written **conjugate Gibbs sampler** (no PyMC/NumPyro dependency; supports Python 3.14+). Produces a per-billet posterior over critical thickness h\_crit, which feeds:

### Adaptive Conformal Prediction — `core/estimator/decision.py`

Adaptive conformal inference maintains **empirical 90% coverage** of h\_crit under distribution shift, requiring only that 1-in-20 billets is audited (discard sectioning or macro-etch, result arriving 5 cycles late).

### Joint Cost Objective — `core/estimator/pipeline.py`

Per-billet cost function solved by `estimator/pipeline.py`:

```
C(h) = c_m · ρ · A_b · h  +  c_d · L_d · P(h < h_crit | data)
```

where `c_m = billet_price_per_kg − remelt_credit_per_kg` (net loss per kg of discard, since discard is remelted, not scrapped).

### Process-Awareness Layer — `skinflow_discard_optimizer/aware/`

| Component | Algorithm | Purpose |
|---|---|---|
| `state_tracker.py` | Particle filter | Online wear / tool-state estimation across cycles |
| `monitor.py` | Hotelling T² + SPE + MEWMA | Multivariate SPC; raises `Predictor.DriftAlarm` |
| `forecast.py` | Forward trajectory model | Projects state evolution for proactive maintenance scheduling |
| `fault_id.py` | Bayesian multi-class classifier | Identifies `die_wear`, `liner_scale`, `lubricant_loss`, etc.; rejects out-of-library faults as `unknown` |

---

## 📁 Repository Layout

```
skinflow_discard_optimizer/      # Main Python package
│
├── sim/                         # Physics-validated digital twin
│   ├── force_model.py           # Ram force: friction + upturn physics
│   ├── defect_model.py          # Skin-flow onset probability model
│   ├── cycle.py                 # Single-billet stroke simulator (1 kHz)
│   ├── faults.py                # Fault / drift injection
│   ├── build_dataset.py         # 200k-cycle dataset builder (seed-based)
│   └── scenarios/               # 13 YAML scenario definitions
│
├── core/                        # Core estimation engine
│   ├── preprocess.py            # Resample · filter · windowing
│   ├── build_features.py        # Feature extraction pipeline
│   ├── features.py              # Functional features (FPCA basis)
│   ├── functional.py            # FPCA fitting
│   ├── store.py                 # Parquet-backed time-series store
│   ├── assembler.py             # Cycle assembler (multi-phase join)
│   ├── safety.py                # SafetyGate: 7 conditions + hard bounds
│   ├── observer/
│   │   ├── ukf.py               # Unscented Kalman Filter (φ-space)
│   │   ├── onset.py             # GLR + BOCPD onset detector
│   │   ├── tune_ukf.py          # UKF hyperparameter tuner (MLE)
│   │   └── eval_onset.py        # Onset detector evaluation harness
│   └── estimator/
│       ├── hierarchical.py      # Conjugate Gibbs sampler (partial pooling)
│       ├── decision.py          # Adaptive conformal prediction intervals
│       ├── fit_decision.py      # Offline calibration
│       └── pipeline.py          # Joint cost objective → h*
│
├── aware/                       # Pattern-awareness layer
│   ├── state_tracker.py         # Particle-filter wear tracker
│   ├── monitor.py               # T² / SPE / MEWMA SPC monitor
│   ├── forecast.py              # Forward trajectory forecaster
│   └── fault_id.py              # Bayesian fault classifier
│
├── eval/                        # Offline evaluation
│   ├── backtest.py              # Rolling-origin backtest
│   ├── ablation.py              # Component ablation study
│   └── stress.py                # Safety-gate stress tests
│
├── service/                     # Streaming service
│   ├── run.py                   # Main OPC-UA subscriber → decision loop
│   └── replay.py                # Parquet-replay mode (no live press)
│
├── roi/                         # ROI aggregation
│   ├── engine.py                # Low / expected / high annual ROI
│   └── dashboard.py             # Combined three-module dashboard
│
├── accel/
│   └── gpu_engine.py            # Optional GPU-accelerated inference path
│
├── config.py                    # Pydantic settings root
└── paths.py                     # Canonical path constants

config/                          # Process & economic YAML configuration
├── press.yaml                   # Press geometry, cycle timings
├── alloys.yaml                  # Alloy-specific flow-stress parameters
├── decision.yaml                # Cost weights, conformal calibration settings
├── defect.yaml                  # Onset model, defect-loss parameters
├── economics.yaml               # Metal price, remelt credit, tariff, rate
├── faults.yaml                  # Fault injection parameters
├── dies.yaml                    # Die geometry
└── coupling.yaml                # DCTO / HPEO coupling parameters

contracts/
└── signals.yaml                 # Full OPC-UA signal contract (all 3 modules)

docs/
├── assumptions.md               # All placeholder values and open questions
├── integration_audit.md         # DCTO + HPEO read-only integration audit
├── kpis.md                      # KPI definitions and measurement methods
└── pitch/                       # Technical pitch documents

tests/                           # 23 pytest test modules
```

---

## 📡 Signal Contract

All inter-module signals are defined in [`contracts/signals.yaml`](contracts/signals.yaml). Key fields per signal:

```yaml
ram_cap_pressure:
  tag:      "ns=2;s=Press01.Ram.CapPressure"
  unit:     bar
  rate_hz:  1000          # 1 kHz — core force-model input
  type:     float
  range:    [0, 380]      # PLACEHOLDER — verify against real press
  owner:    skinflow
  kind:     measured
```

**Signals the plan needs that neither DCTO nor HPEO currently provide:**
- Cap pressure and rod pressure (separate, 1 kHz)
- Container liner temperatures 1–4
- Billet temperature, oil temperature
- Supply-pressure minimum during stroke
- Butt shear position

These are defined fresh in `contracts/signals.yaml` with `owner: skinflow` and must be mapped to real PLC tag names during integration.

**SKDO output tags** (`owner: skinflow`, `kind: derived`):

| Tag | Meaning |
|---|---|
| `Predictor.ButtCutMm` | Recommended discard cut (mm) |
| `Predictor.ButtCutIntervalLo/Hi` | 90% conformal prediction interval bounds |
| `Predictor.OnsetDepthMm` | Detected skin-flow onset depth |
| `Predictor.DriftAlarm` | Bool — T²/MEWMA drift alarm active |
| `Predictor.FaultClass` | String — identified fault or `healthy`/`unknown` |
| `Predictor.ConfidenceLevel` | Enum — `HIGH` / `MEDIUM` / `LOW` / `CRITICAL_FALLBACK` |

---

## ⚙️ Configuration

All constants live in versioned YAML files under `config/`. **All values marked `PLACEHOLDER` must be replaced with sourced plant data before production use.**

| File | Key contents |
|---|---|
| `press.yaml` | Container bore, billet geometry, ram area, stall limit, phase timings |
| `alloys.yaml` | Flow-stress σ₀ and n per alloy (AA6063, AA6082); taper and deformation-heating coefficients |
| `decision.yaml` | Conformal target coverage, audit rate, audit delay, cost weights |
| `defect.yaml` | Onset amplitude threshold, Weibull defect model parameters |
| `economics.yaml` | `billet_price_per_kg`, `remelt_credit_per_kg`, `defect_loss_per_event`, tariff, press rate |
| `coupling.yaml` | DCTO shear-stroke timing (source: `DCTO/simulator/press_config.py`) |
| `faults.yaml` | Fault injection parameters for all 13 scenarios |

---

## 📊 Evaluation Results (Phase 5)

Evaluated on a **rolling-origin backtest** across **13 scenarios** (11 development + 2 fully held-out) spanning **12,660 billet cycles** from the physics-validated digital twin. Source: [`eval/backtest.py`](skinflow_discard_optimizer/eval/backtest.py).

### Headline KPIs

| KPI | Point Estimate | 95% CI | Target | Status |
|---|:---:|:---:|:---:|:---:|
| Mean discard thickness | **29.5 mm** | [28.6, 30.2] | < 40 mm static | ✅ PASS |
| Metal recovery vs. static | **1.29% billet mass** | [1.20%, 1.40%] | > 0 | ✅ PASS |
| Defect rate | **0.56 / 1,000 billets** | [0.44, 0.68] | ≤ static baseline | ⚠️ CAUTION |
| False-alarm rate | **4.9 / 1,000 cycles** | [2.9, 7.4] | ≤ 5.0 / 1,000 | ✅ PASS |
| Warning lead time (median) | **300 cycles (~7 h)** | [−29, 1,546] | median > 0 | ✅ PASS |
| Conformal interval coverage | **91.4%** | [90.9%, 92.1%] | 90% ± 2% | ✅ PASS |
| Cut-decision latency (p99) | **15.8 ms** | — | < 200 ms | ✅ PASS |
| Net metal saving / billet | **0.390 EUR** | [0.346, 0.463] | > 0 | ✅ PASS |

> **Defect rate ⚠️ CAUTION:** SKDO introduces a small positive defect risk relative to the near-zero rate of the conservative static-cut baseline. The absolute rate (0.56/1,000) is well below typical industry total-reject rates of 5–15/1,000.

### Per-Scenario Breakdown

| Scenario | Held-out | Cycles | Mean cut (mm) | Recovery (%) | Defect/1k | Saving (EUR/billet) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| healthy_baseline | — | 1,260 | 28.4 | 1.42 | 0.59 | 0.442 |
| alloy_change | — | 540 | 28.6 | 1.40 | 0.44 | 0.486 |
| cold_start_days | — | 540 | 28.6 | 1.41 | 0.51 | 0.462 |
| die_wear | — | 540 | 31.1 | 1.09 | 0.50 | 0.321 |
| encoder_offset | — | 540 | 29.2 | 1.33 | 0.11 | 0.588 |
| flash_spike | — | 540 | 28.5 | 1.41 | 0.53 | 0.457 |
| liner_scale | — | 540 | 32.6 | 0.90 | 0.55 | 0.211 |
| lubricant_loss | — | 540 | 28.6 | 1.40 | 0.82 | 0.340 |
| sensor_gain_drift | — | 540 | 28.5 | 1.41 | 0.37 | 0.521 |
| supply_pressure_sag | — | 540 | 28.2 | 1.45 | 0.41 | 0.522 |
| temperature_drift | — | 540 | 26.9 | 1.60 | 0.58 | 0.531 |
| **combined_wear_and_scale** | **✅** | 3,200 | 30.8 | 1.13 | 0.47 | 0.347 |
| **die_change** | **✅** | 2,800 | 29.2 | 1.32 | 0.80 | 0.310 |

### Drift Detection Lead Times

| Scenario | Lead Time | Primary Channel | Status |
|---|:---:|:---:|:---:|
| die_wear | +222 cycles | MEWMA | ✅ |
| liner_scale | −202 cycles | T² | ⚠️ |
| temperature_drift | +4,660 cycles | MEWMA | ✅ |
| lubricant_loss | +66 cycles | T² | ✅ |
| supply_pressure_sag | +300 cycles | MEWMA | ✅ |
| sensor_gain_drift | −29 cycles | T² | ⚠️ |
| encoder_offset | +469 cycles | MEWMA | ✅ |
| flash_spike | +1,546 cycles | MEWMA | ✅ |
| combined_wear_and_scale | +1,406 cycles | MEWMA | ✅ |

> **⚠️ CAUTION scenarios** (`liner_scale`, `sensor_gain_drift`) involve abrupt step-changes rather than gradual drift, which outpace the monitor's response window.

---

## 🛡️ Safety Gate

[`core/safety.py`](skinflow_discard_optimizer/core/safety.py) applies 7 conditions before any `Predictor.ButtCutMm` is published:

1. Telemetry validator passes (all required signals in-range, no frozen values, no timestamp jitter)
2. Alloy ID is in the calibrated library
3. Model confidence is `≥ MEDIUM`
4. Recommended cut is within hard bounds **[12, 60] mm**
5. Posterior standard deviation is below the configured maximum
6. Fault classifier is not reporting an out-of-library (`unknown`) fault above threshold
7. Conformal interval width is below the maximum permitted width

If **any condition fails**, the gate issues `CRITICAL_FALLBACK` and clamps `Predictor.ButtCutMm = 40.0 mm` (the conservative static rule).

**Stress test results** (`eval/stress.py`) — 7/7 scenarios correctly trigger CRITICAL_FALLBACK:

| Stress Scenario | h_cut_mm | Fallback |
|---|:---:|:---:|
| missing_samples | 40.0 | ✅ |
| frozen_sensor | 40.0 | ✅ |
| timestamp_jitter | 40.0 | ✅ |
| out_of_range pressure | 40.0 | ✅ |
| unseen_alloy (AA7075) | 40.0 | ✅ |
| novel_fault (OOL) | 40.0 | ✅ |
| heavy_drift | 40.0 | ✅ |

---

## 📏 KPIs

Money is never a standalone KPI. The ROI layer converts evaluation-interval KPIs using `config/economics.yaml` and always reports **Low / Expected / High**.

Per-billet cost:
```
C(h) = c_m · ρ · A_b · h  +  c_d · L_d · P(h < h_crit | data)
```

Supporting metrics reported alongside headline KPIs:
- **Interval coverage** — fraction of cycles where true h\_crit lies inside the published interval (target: 90% ± 2%)
- **Onset-position error** — mm between detected and true skin-flow onset
- **Fault-identification accuracy** — confusion matrix over injected faults, including `unknown` class
- **Cut-decision latency** — p99 wall time from end-of-stroke data to `Predictor.ButtCutMm` published (target: < 200 ms)

Full KPI definitions and measurement methods: [`docs/kpis.md`](docs/kpis.md)

---

## ⚠️ Integration Hazards

The full audit is in [`docs/integration_audit.md`](docs/integration_audit.md). Key conflicts between DCTO and HPEO that SKDO resolves:

| # | Conflict | Resolution |
|---|---|---|
| C1 | Phase names differ (DCTO: 8 phases, HPEO: 7) | Canonical enum in `contracts/signals.yaml` with alias maps from both |
| C5 | DCTO names the ram-position tag `ContainerPosition` | Contract separates `ram_position` and `container_position` |
| C6 | DCTO and HPEO both bind `opc.tcp://127.0.0.1:4840` | SKDO subscribes only; mock mode uses port **4842** |
| C7 | Both have top-level `simulator/` and `opcua/` packages | Everything here lives under `skinflow_discard_optimizer/`; sibling repos never added to `sys.path` |
| C9 | DCTO pins `numpy<2.0`; this environment uses numpy 2.x | SKDO targets numpy 2.x; never imports DCTO code at runtime |
| C11 | HPEO writes pump staging setpoints | SKDO is advisory only; writes only `Predictor.*` |

---

## 🧪 Pilot Programme Required

Results are from simulation. The following must be verified on real data before commercial use:

1. **Force-curve shape calibration** — The digital twin uses a friction-based upturn. Real presses may show a pressure-drop shape. The GLR detector supports both; the `direction` parameter must be set from real force curves.
2. **Alloy-specific model fitting** — The hierarchical Bayesian model covers AA6063 and AA6082. Each additional alloy requires ~200–500 calibration billets.
3. **Defect labelling** — `h_crit` is simulated. A 3–6 month data collection period with matched quality-gate records is required to calibrate onset detection against real contamination events.
4. **OPC-UA signal mapping** — Tag names and engineering units must be mapped against `contracts/signals.yaml` for each specific press controller.
5. **Latency budget verification** — p99 < 200 ms is confirmed under simulation load. Real OPC-UA sampling jitter and network round-trip must be measured.
6. **Critical hypothesis** — Onset detection is only actionable if the end-of-stroke force change starts *before* the cost-optimal cut point. The simulator places onset 10 ± 0.8 mm above h\_crit. **Measuring this gap on real force curves is the first thing a pilot must do.**

---

## 🚀 Installation & Running

**Requirements:** Python ≥ 3.11, no GPU required (GPU engine in `accel/gpu_engine.py` is optional).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install the package and dependencies
pip install -e .

# 3. (Optional) install dev dependencies
pip install -e ".[dev]"
```

**Generate the simulation dataset:**
```bash
python -m skinflow_discard_optimizer.sim.build_dataset
```

**Run the offline backtest:**
```bash
python -m skinflow_discard_optimizer.eval.backtest
```

**Run the streaming service (replay mode — no live press):**
```bash
python -m skinflow_discard_optimizer.service.replay
```

**Run the streaming service (live OPC-UA mode):**
```bash
python -m skinflow_discard_optimizer.service.run
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run without slow tests
pytest -m "not slow"

# Run a specific module
pytest tests/test_ukf.py -v
```

23 test modules covering: UKF correctness, onset detection, conformal calibration, decision pipeline, defect model, force model, backtest, ablation, stress / safety gate, contracts/validation, ROI engine, service, state tracker, monitor, forecast, fault identification, GPU engine, and functional/FPCA.

---

## 📋 Key Assumptions

All placeholder values and judgement calls are documented in [`docs/assumptions.md`](docs/assumptions.md). The most critical:

| # | Assumption | Status |
|---|---|---|
| A-02 | PLC signal tag names are from `contracts/signals.yaml` — not from any real tag list | **Must verify against real press** |
| A-03 | Signal `range` fields are sanity limits, not process limits | **PLACEHOLDER — widen/narrow per press** |
| A-06 | All values in `config/economics.yaml` are PLACEHOLDER | **Must replace with sourced plant data** |
| A-12 | Force upturn leads skin flow by ~10 mm (key hypothesis) | **Must measure on real force curves first** |
| A-26 | Conformal calibration assumes 1-in-20 billets audited, result 5 cycles late | **Must agree audit protocol with plant** |

---

## 🚫 Limitations

- **Does not control any actuator.** Publishes advisory tags only.
- **Does not guarantee zero defects.** Reduces expected discard while keeping defect risk low; a residual rate of ~0.56/1,000 exists (vs ~0/1,000 static rule).
- **Does not operate on unseen alloys without calibration.** SafetyGate issues CRITICAL_FALLBACK for alloy IDs not in the calibrated library.
- **Does not replace operator judgement.** Every `Predictor.*` signal is a recommendation the operator accepts or overrides.
- **Conformal guarantee degrades without audits.** Without periodic discard sectioning, the prediction interval is only as good as its offline calibration.

---

## 📄 License

_License TBD — contact the repository owner._

---

*Part of the **Press Value Platform** — SKDO (this repo) · [DCTO](../DCTO) · [HPEO](../HPEO)*
