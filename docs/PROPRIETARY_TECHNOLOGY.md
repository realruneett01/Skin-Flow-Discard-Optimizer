# Proprietary technology inventory

**Confidential. Internal use only.**

This inventory comes from reading the code in this repository. It separates three things that are easy to blur:

1. **Building blocks** that are published methods. They are not secrets, and nobody here should claim to have invented them.
2. **Our contributions**: design decisions, reformulations, integrations and tests that exist in this code. Whether they are unavailable elsewhere is a claim someone has to check (see "What this document cannot tell you").
3. **Assets**: data, calibrations and artifacts that are ours by construction and are the real moat.

## 1. Building blocks (public, do not claim)

| Used here | Where |
|---|---|
| Sellars–Tegart flow stress, Zener–Hollomon parameter, Feltham strain rate | `sim/force_model.py` |
| Unscented Kalman filter | `core/observer/ukf.py` |
| Generalised likelihood ratio test, Bayesian online change-point detection | `core/observer/onset.py` |
| Hierarchical (partial-pooling) Bayesian regression, Gibbs sampling | `core/estimator/hierarchical.py` |
| Split and adaptive conformal prediction | `core/estimator/decision.py` |
| Functional PCA, Hotelling T², SPE/Q, MEWMA | `core/functional.py`, `aware/monitor.py` |
| Particle filter with regime switching, Savitzky–Golay, Lepski window selection | `aware/state_tracker.py`, `core/preprocess.py` |

## 2. Our contributions (verify before asserting novelty)

| # | Contribution | What is specific to us | Evidence in repo |
|---|---|---|---|
| C1 | **φ-space reparameterisation of the stroke observer.** The ram-force model is bilinear in (flow-stress scale s, friction μ). The filter state is φ = [s, s·μ, F_tool], which makes the measurement linear, so the unscented update is exact. | A direct UKF in θ was tested and drifted from the exact posterior (the posterior is a hyperbolic ridge). | `ukf.py` docstring and `docs/assumptions.md` A-23 |
| C2 | **Closed-form onset posterior from the GLR sums.** The same running sums that drive the alarm give a posterior over the *position* of skin-flow onset, marginalised over the decay length, in O(grid) per sample. | The onset is defined at a reference amplitude (A-19), not at the alarm. | `onset.py` |
| C3 | **Cost-optimal cut with a closed-form optimum and probit-smoothed defect transition.** The optimum h* has an analytic solution including the logistic clean-to-defective transition width. | Extended to a joint objective with dead-cycle time and pump-energy terms that keeps the closed form. | `decision.py`, `joint.py` |
| C4 | **Causal, commit-aware decision engine.** The engine replays a stroke in time order, commits when the ram is within a latency margin of the recommendation, flags `late` decisions, and uses nothing after the commit point. | Training features come from the same code as live (`features_at`), which removes train/serve skew. | `pipeline.py` |
| C5 | **Exact batched GPU engine.** The per-stroke UKF + GLR + BOCPD loop is re-expressed as padded tensor ops (cumulative sums, batched 3×3 solves). It runs in float64 and is tested against the CPU engine. | It speeds up fitting and evaluation without a second algorithm to trust. | `accel/gpu_engine.py`, `tests/test_gpu_engine.py` |
| C6 | **Digital twin with exact ground truth.** Every simulated cycle carries its true h_crit, an oracle cut, and the true cost of any cut. The latent state drives both force and h_crit, so the signal is informative but imperfect. | This lets the system report regret against an oracle, not just error. Includes 13 scenarios with 2 fully held-out. | `sim/defect_model.py`, `sim/scenarios/`, `eval/backtest.py` |
| C7 | **Pattern awareness stack coupled to the cut.** Latent wear/scale/taper/sensor-gain tracking, forward Monte-Carlo time-to-limit, and a Bayesian fault classifier with an out-of-library reject class (`unknown`) that feeds the safety gate. | The classifier output changes whether the cut is published or falls back. | `aware/*`, `core/safety.py` |
| C8 | **Safety gate with a hard advisory contract.** Seven-condition validation, hard bounds [12, 60] mm, conservative 40 mm fallback, outputs confined to `Predictor.*`. The contract schema rejects any output outside that namespace and any input that uses it. | Verified by seven stress scenarios that all end in a bounded, logged fallback. | `safety.py`, `contracts/schema.py`, `eval/stress.py` |
| C9 | **Provenance-enforced configuration.** Every numeric parameter must carry `status` (PLACEHOLDER or SOURCED) and `SOURCE`, or loading fails. | No unsourced constant can reach a model or an ROI figure unmarked. | `config.py` |
| C10 | **Layer-by-layer ablation.** Each of the seven layers must show a measured contribution or be removed. | Tells you which parts earn their keep. | `eval/ablation.py` |
| C11 | **Three-module value platform on one tag stream** (SKDO, DCTO, HPEO) with a shared signal contract and low/expected/high ROI with intervals. | DCTO and HPEO live in other repositories; only SKDO's side is here. | `contracts/signals.yaml`, `roi/engine.py` |
| C12 | **Machine-bound deployment gate and portable bundle.** A preflight that must pass on *this* machine with *this* config, contract and models before the service starts. It includes a loopback test proving the live OPC-UA path reproduces the direct-replay cut. The bundle is self-contained and hash-verified, and sticky model tiers let prebuilt models travel. | Engineering integration rather than a novel algorithm; useful as a commercial control more than as a patent claim. | `deploy/preflight.py`, `deploy/bundle.py` |

## 3. Assets (ours by construction)

| Asset | Status |
|---|---|
| Fitted model artifacts: `fpca.npz`, `hcrit_models.npz`, `conformal_scores.npy`, `ukf_params.json`, `monitor_config.npz` | Gitignored, rebuilt by `skdo bootstrap`. They are the product of C1–C6 on a given press configuration. |
| Calibrations in `config/*.yaml` | **Today 133 of 135 parameters are PLACEHOLDER** (2 SOURCED), derived from assumptions and literature, not measured on a press. Real press data would turn this into the strongest asset. |
| Real force-curve data from pilot presses | Does not exist yet. A-12 (force upturn leads skin flow by about 10 mm) is the hypothesis everything depends on. |
| Scenario library and fault signatures | Simulation-based today. |

## 4. What this document cannot tell you

- **Whether anything here is "not available outside."** I have not searched patents, papers, or vendor products, and a code read cannot show that. C1, C2, C4 and C7 are the most plausible candidates for a defensible claim. Before relying on that, run a prior-art search and talk to an IP attorney. Some of this may be better protected as trade secret than patent. Patent filing generally requires disclosure. You said you want to make the repositories private, so they are presumably public today: anything already pushed has been exposed, which can weaken both trade-secret and patent positions (some jurisdictions give a short grace period after your own disclosure, others none). Raise this with the attorney.
- **Whether it works on a real press.** Every performance number comes from the digital twin (see `docs/pitch/04_assumptions_and_limits.md`).

## 5. Protecting it

- Keep the repository private, with no public forks. Check Settings → Danger zone → visibility. Do not remove history to "hide" earlier exposure: forks, clones and caches may exist.
- `LICENSE` is "all rights reserved". `pyproject.toml` carries `Private :: Do Not Upload`, which makes PyPI reject an accidental upload.
- Models, data and site config stay out of git and out of the Docker image (`.gitignore`, `.dockerignore`). They live in volumes.
- Publish the image only to a private registry. The CI workflow pushes to GHCR under the repository's own (private) visibility.
