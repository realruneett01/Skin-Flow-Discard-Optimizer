# Press Value Platform — Architecture Diagram

> **One page.** Every block below maps 1-to-1 to a source file in the repository.

---

## System Overview

```mermaid
flowchart TD
    OPC["OPC-UA Tag Stream\n(contracts/signals.yaml)"]

    subgraph P2["Phase 2 — Signal & Features"]
        PRE["Preprocessing\ncore/preprocess.py"]
        FPCA["FPCA Basis\ncore/functional.py"]
        FE["Feature Extractor\ncore/features.py"]
    end

    subgraph P3["Phase 3 — Core Engine"]
        UKF["Within-Stroke UKF\ncore/observer/ukf.py"]
        ONS["GLR+BOCPD Onset Detector\ncore/observer/onset.py"]
        HB["Hierarchical Bayes + Conformal\ncore/estimator/pipeline.py"]
        JO["Joint Objective\n(yield / dead-cycle / pump energy)\ncore/joint.py"]
    end

    subgraph P4["Phase 4 — Pattern Awareness"]
        PF["Particle-Filter Wear Tracker\naware/state_tracker.py"]
        MON["T² / SPE / MEWMA Monitor\naware/monitor.py"]
        FC["Bayesian Fault Classifier\naware/fault_id.py"]
        FCT["Forward Trajectory Forecaster\naware/forecast.py"]
    end

    subgraph P5["Phase 5 — Safety Gate"]
        SG["SafetyGate\n7-condition validator\ncore/safety.py"]
    end

    subgraph SVC["Phase 6 — Streaming Service"]
        RUN["StreamingOptimizerService\nservice/run.py"]
    end

    subgraph OUT["Advisory Outputs — Predictor.*"]
        B["ButtCutMm"]
        LO["CutIntervalLowMm"]
        HI["CutIntervalHighMm"]
        CF["CutConfidence"]
        DA["DriftAlarm"]
        FCC["FaultClass"]
        FB["IsFallback"]
    end

    subgraph ROI["Phase 6 — ROI Layer"]
        ENG["ROI Engine\nroi/engine.py"]
        DASH["Streamlit Dashboard\nroi/dashboard.py"]
    end

    OPC --> PRE
    PRE --> FPCA
    FPCA --> FE
    FE --> UKF
    UKF --> ONS
    ONS --> HB
    HB --> JO
    JO --> SG
    FE --> PF
    FE --> MON
    FE --> FC
    PF --> FCT
    FC --> SG
    MON --> SG
    SG --> RUN
    RUN --> B & LO & HI & CF & DA & FCC & FB
    B & LO & HI --> ENG
    ENG --> DASH
```

---

## Data Flow Per Billet Cycle

```mermaid
sequenceDiagram
    participant OPC as OPC-UA Tags
    participant P2 as Phase 2 Features
    participant P3 as Phase 3 Engine
    participant P4 as Phase 4 Awareness
    participant SG as Safety Gate
    participant HMI as Operator HMI

    OPC->>P2: Raw stroke arrays (t, x, p_cap, p_rod)
    P2->>P3: Resampled + FPCA feature vector
    P3->>P3: UKF filter → onset detection → conformal decision
    P3->>SG: CycleDecision (h_cut_mm, confidence)
    P2->>P4: Feature vector → monitor update
    P4->>SG: drift_alarm, fault_class
    SG->>SG: 7-condition validation + hard bounds [12, 60] mm
    SG->>HMI: Predictor.ButtCutMm [L, E, H] + 6 advisory tags
    Note over SG,HMI: p99 latency < 200 ms
    Note over HMI: Operator accepts or overrides — no setpoint write
```

---

## Safety Fallback Logic

```mermaid
flowchart LR
    IN["CutCandidate\n(raw_cut_mm, confidence,\nfault_class, telemetry_issues)"]
    V1{"Telemetry\nvalid?"}
    V2{"Process context\nvalid?"}
    V3{"Confidence\nhigh/medium?"}
    V4{"Within hard\nbounds?"}
    OK["GuardedDecision\nis_fallback=False"]
    FB["GuardedDecision\nis_fallback=True\nh_cut_mm=40.0 mm\nalert=CRITICAL_FALLBACK"]

    IN --> V1
    V1 -- "PASS (7 checks)" --> V2
    V1 -- "FAIL" --> FB
    V2 -- "PASS" --> V3
    V2 -- "FAIL" --> FB
    V3 -- "PASS" --> V4
    V3 -- "FAIL" --> FB
    V4 -- "PASS" --> OK
    V4 -- "FAIL → clamp" --> FB
```

---

## Module Mapping

| Module | Phase | Primary file | Advisory tag |
|---|---|---|---|
| Skin-Flow Discard Optimizer (SKDO) | 3 | `core/estimator/pipeline.py` | `Predictor.ButtCutMm` |
| Dead-Cycle Timer Optimizer (DCTO) | 3 | `core/joint.py` | (throughput advisory) |
| Hydraulic Pump Energy Optimizer (HPEO) | 3 | `core/joint.py` | (pressure-schedule advisory) |
| Wear-State Tracker | 4 | `aware/state_tracker.py` | `Predictor.DriftAlarm` |
| Fault Classifier | 4 | `aware/fault_id.py` | `Predictor.FaultClass` |
| Safety Gate | 5 | `core/safety.py` | `Predictor.IsFallback` |
| ROI Engine | 6 | `roi/engine.py` | (dashboard only) |
