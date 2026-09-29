# KPIs

Task 0.3. These are the headline numbers every evaluation (Phase 5) must report, each with an interval. Targets are **PLACEHOLDER** until agreed with the plant.

| KPI | Unit | Definition | How it is measured | Target (PLACEHOLDER) |
|---|---|---|---|---|
| Butt (discard) thickness | mm | Thickness of metal left in the container and sheared off, `h_cut` | Mean and distribution over cycles; compared with the static conservative cut | Lower than the static cut, with no rise in defect rate |
| Recovery | % of billet mass | `(h_static - h_cut) * A_b * rho / m_billet * 100`, averaged over cycles | Per-cycle from the recommended cut; negative when the model cuts thicker than static | Reported with bootstrap CI; no fixed target |
| Defect rate | defects per 1000 billets | Cycles where `h_cut < h_crit` (skin-flow material enters the extrudate) | Exact in simulation (true `h_crit` is known); from quality records in a pilot | Not above the static-cut baseline |
| False-alarm rate | alarms per 1000 cycles | Drift alarms raised on cycles with no active fault | Healthy scenarios only | <= 5 per 1000 cycles |
| Warning lead time | cycles | Cycles between the first alarm and the cycle where the fault first causes a limit breach (defect probability above limit, or cut outside bounds) | Injected-fault scenarios; negative if the alarm comes after the breach | Reported as a distribution; median > 0 |
| Cut-decision latency | ms | Wall time from end-of-stroke data available to `Predictor.ButtCutMm` published | Measured in the streaming service (Task 6.1) | p99 < 200 ms |

Supporting metrics (reported, not headline):

- **Interval coverage:** fraction of cycles whose true `h_crit`-optimal cut lies inside the published interval. Target: within 2 points of nominal (Task 3.3).
- **Onset-position error:** mm between detected and true skin-flow onset (Task 3.2).
- **Fault-identification accuracy:** confusion matrix over injected faults, including the `unknown` class (Task 4.4).

## Money

Money is never a KPI on its own. The ROI layer (Task 6.2) converts the KPIs above with `config/economics.yaml` and always reports **low / expected / high**, where low and high come from the Phase 5 intervals.

Per-billet cost used by the decision layer (Section 1.2 of the plan):

```
C(h) = c_m * rho * A_b * h  +  c_d * L_d * P(h < h_crit | data)
```

with `c_m = billet_price_per_kg - remelt_credit_per_kg` (the value actually lost per kg of discard, since the discard is remelted) and `c_d * L_d = defect.loss_per_event`.
