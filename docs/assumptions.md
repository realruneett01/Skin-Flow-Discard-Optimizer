# Assumptions and open questions

Every placeholder, inferred value and judgement call made while building this repo. Each entry says what was assumed, why, and what would replace it. Update this file whenever an assumption is confirmed or removed.

## Plan placeholders

| Placeholder | Value used | Status |
|---|---|---|
| `[DCT_REPO]` | `../DCTO` (Dead-Cycle Time Optimizer) | Inferred from the folder name and README. **Confirm.** |
| `[PUMP_REPO]` | `../HPEO` (Hydraulic Pump Energy Optimizer) | Inferred from the folder name and README. **Confirm.** |

## Assumptions

**A-01. Sibling repos are read-only references.** This repo never imports DCTO or HPEO code at runtime. They share package names (`simulator`, `opcua`) and DCTO pins `numpy<2` (audit C7, C9). Values we need from them are copied into `config/` with a `SOURCE:` naming the file.

**A-02. Signals neither project has.** Separate cap and rod pressure at 1 kHz, liner temperatures 1-4, billet temperature, oil temperature, measured supply pressure and shear position are defined in `contracts/signals.yaml` from the plan, not from any existing tag list. Real tag names must come from the press PLC.

**A-03. Contract ranges are sanity limits.** The `range` on each signal rejects corrupt data (for example a 900 °C billet or a negative position). The ranges are PLACEHOLDERs and are wider than the process window.

**A-04. Decision cost uses the net metal loss.** The plan's `c_m` is taken as `billet_price_per_kg - remelt_credit_per_kg`, because the discard goes back to the cast house and is not lost outright. Using the full billet price would overstate the value of a thinner cut by roughly 5x with the current placeholders. *Flagged under plan rule 7 as an interpretation, not a formula change.*

**A-05. No viscosity model in HPEO.** Task 1.3 says to import the viscosity model from `[PUMP_REPO]` if available. HPEO has no oil temperature or viscosity code, so the supply-pressure sag uses our own placeholder model (see `config/faults.yaml` when it exists).

**A-06. Economics are all placeholders.** Every value in `config/economics.yaml` is PLACEHOLDER, including those copied from DCTO, because DCTO does not cite sources for them either.

**A-07. Environment.** Development is on Python 3.14 with numpy 2.x in a project `.venv`. Libraries without Python 3.14 wheels (possibly PyMC or NumPyro) will be replaced with an equivalent written on NumPy/SciPy. Each case will be noted here.

**A-08. Currency.** EUR, matching both existing projects.

**A-09. Mock OPC-UA port.** Both sibling servers bind port 4840 (audit C6). This repo's mock server uses 4842.

## Phase 1 (simulator)

**A-10. Force formula checks (plan rule 7).** The plan's force formula was implemented as written, with these readings, all flagged rather than changed:
- `Db` is the container bore, because the billet is upset to fill the container. `L0` is the upset length, found by conserving volume.
- `De = Db / sqrt(R)` (equivalent round diameter).
- The formula has no redundant-work term (Johnson's `a + b ln R`), so it underestimates the absolute force of a real die. `sigma_scale` (theta[0]) absorbs this, so the onset logic is unaffected.
- Sign conventions are correct: friction falls as the ram advances, and `F_up` grows as `h -> 0`.
- `mu` is interpreted through sticking friction (`mu = m / sqrt(3)`), giving about 0.55 instead of a Coulomb coefficient.

**A-11. Simulator-only force terms.** The simulator adds a container-fill ramp (first ~3 mm) and a dummy-block entry/breakthrough bump (peak ~8 mm, gone within ~5% of the stroke). They are not in the plan formula. They exist so the Task 2.1 gate can be tested against a real start transient.

**A-12. KEY HYPOTHESIS: the force change leads skin flow.** Onset detection can only help if the end-of-stroke force change starts *before* the cost-optimal cut point, because the press has to extrude past the onset to see it. With the placeholder economics the oracle cuts about 6.8 mm above `h_crit`, so the simulator places the onset `10 ± 0.8 mm` above `h_crit`. If a real press shows the change only a few mm before (or after) `h_crit`, onset detection cannot drive the cut, and the value has to come from cross-cycle prediction (layers L5-L6) alone. **Measuring this gap on real force curves is the first thing a pilot must do.**

**A-13. Causal decisions.** The simulator runs every stroke to 8 mm remaining so the whole curve is known. Every decision and evaluation must use only the samples recorded before the ram reaches the recommended cut (`h > h*`). The backtest enforces this.

**A-14. Scenario count.** The plan asks for 12 scenarios, but its own list comes to 13 (1 healthy + 8 single faults/drifts + 1 combined + cold start + alloy change + die change). All 13 are built.

**A-15. Dataset stores seeds, not raw strokes.** 200k strokes at 1 kHz would be ~13 billion samples. `cycles.parquet` stores each cycle's stroke parameters and `stroke_seed`, and `regenerate_stroke(row)` rebuilds the exact 1 kHz stream on demand. Rows in the fast path compute pump energy on a 50 Hz noise-free pass, which is within 0.2% of the 1 kHz value.

**A-16. Static cut is safe under these placeholders.** With the current coefficients the 40 mm static cut produces essentially zero defects in every scenario. The value on offer is metal recovery (oracle ceiling ~0.64 EUR/billet). Stress tests in Task 5.3 push drift beyond the library to check that the model stays safe where the static cut would not.

## Phase 2 (signal and features)

**A-17. Time-series DB.** Phase 0 did not define one. Features go to a Parquet-backed store (`core/store.py`, one file per press or scenario, upsert by cycle) behind a `TimeSeriesStore` interface, so an InfluxDB or TimescaleDB adapter can replace it without touching the feature code.

**A-18. Ram stall.** The plan's exponential upturn can demand more force than the press has. The simulator now ends the stroke where the required force exceeds what the 300 bar supply can deliver (`stall_limit_N`), as a real press would stall. This changed some strokes on worn dies.

**A-19. Onset is a defined point, not "first departure".** `F_up = a*exp(-h/lam)` has no sharp start, so "where it leaves the noise" depends on the noise level. The onset is defined as the thickness where `F_up` reaches `upturn.amplitude_at_onset_N` (0.25 MN), both in the simulator and in the estimators. Because the rise is visible before that point, the onset can be extrapolated from data that stops above it.

**A-20. Fitted theta uses the nominal taper.** The per-cycle least-squares theta uses the measured billet front temperature with the nominal taper and deformation heating (not measured per billet). F_tool reads ~5% low on healthy strokes as a result; `sigma_scale` and `mu` are within 1%.

**A-21. Derivative windows are chosen per position.** One global Savitzky-Golay window cannot serve both the near-linear mid-stroke and the sharp end-of-stroke bend, so the window is chosen per position from the noise level (Lepski's method, kappa 3.5, windows 2.1-40 mm). Measured error against the noise-free truth: dF/dx about 2% mid-stroke and under 1% in the gated tail; d2F/dx2 about 5% in the tail.

**A-22. Shear phase attribution.** In DCTO's phase order the shear that cuts billet n's discard runs at the start of cycle n+1. The assembler attributes each phase to the cycle it occurs in. The joint model (Task 3.4) uses the expected shear time as a function of the cut, so this does not bias it.

## Phase 3 (core engine)

**A-23. The UKF filters phi = [s, s*mu, F_tool], not theta.** The force model is bilinear in theta (the `s*mu` product), and the data pin `s*mu` down far more tightly than `s` alone, so the posterior in theta is a curved ridge that a Gaussian cannot represent. A UKF run directly on theta drifted well away from the exact posterior (s = 0.87 against 1.07 on the same data). In phi the measurement is linear, so the unscented update is exact. theta and its covariance are recovered by an unscented transform. The plan's "UKF over theta" interface is kept.

**A-24. UKF details found by testing.**
- Ram speed feeds the flow stress. A 0.25 s two-point speed estimate (~1% noise) biased theta through errors-in-variables, so speed is now the least-squares slope over the past 1 s.
- The filter starts at 12% of stroke, not at the Task 2.1 transient end (~5%). The few-kN tail of the breakthrough bump still biases the weakly separable sigma_scale/F_tool pair.
- Maximum likelihood drives the process noise to ~0 on healthy strokes (theta really is constant within a stroke).
- Theta is scored against the *effective* truth: the filter uses the measured billet temperature (1 degC sensor noise), so its exact sigma_scale is sigma(T_true)/sigma(T_measured), not 1.
- sigma_scale and F_tool remain weakly identifiable individually. mu and the force prediction are solid.

**A-25. No PyMC/NumPyro.** Neither supports Python 3.14 in this environment. The hierarchical model (partial pooling by die and alloy) is a hand-written conjugate Gibbs sampler (`core/estimator/hierarchical.py`), which is exact for this model.

**A-26. Conformal calibration needs audited labels.** Adaptive conformal inference only keeps coverage under drift if true `h_crit` values come back. A real plant does not observe `h_crit` per billet. The evaluation assumes 1 billet in 20 is audited (discard sectioning or macro-etch) and the result arrives 5 cycles late (`config/decision.yaml`). Without audits the interval is only as good as its offline calibration.

**A-27. Physics-informed onset model: predict the gap.** The onset model predicts `h_crit - onset`, so the onset enters with coefficient 1 (hypothesis A-12: the upturn leads skin flow by a roughly constant gap). A free regression coefficient came out ~0.5 on mostly healthy training data and under-followed drift (liner-scale interval coverage 50%). The cost is slightly lower precision on healthy strokes (residual sd ~1.2 mm vs 0.94 mm). The onset is used only once its posterior sd is below 2 mm; earlier, the wide extrapolation forced premature, thick cuts. The regressors are mu, dT and billet temperature only; sigma_scale and F_tool (weak ridge, A-24) gave meaningless coefficients.

**A-28. Decisions and scoring are in press coordinates.** The press stops the ram on its *measured* position. With an encoder offset the true cut differs from the recorded number by the offset, so evaluation converts cuts and intervals to true thickness before scoring. The engine itself is self-consistent: onset and cut are both measured in the same, offset, coordinates.
