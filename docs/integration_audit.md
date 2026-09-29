# Integration audit: Dead-Cycle Timer Optimizer (DCTO) and Hydraulic Pump Energy Optimizer (HPEO)

Task 0.1. Read-only audit of `../DCTO` (`[DCT_REPO]`) and `../HPEO` (`[PUMP_REPO]`). No code in either repo was modified.

Audited at DCTO commit `e771e0c` and HPEO commit `60ec80c`.

## 1. Summary per project

| | DCTO (Dead-Cycle Timer Optimizer) | HPEO (Hydraulic Pump Energy Optimizer) |
|---|---|---|
| Language | Python (README targets 3.11) | Python |
| Libraries | asyncua, **numpy >=1.26,<2.0**, pandas, scipy, ruptures, hmmlearn, streamlit, plotly, matplotlib, pytest, pytest-asyncio | numpy >=1.26, pandas, scipy, scikit-learn, streamlit, plotly, matplotlib, asyncua, requests, pytest |
| Input signals | Phase durations per cycle (event-level, one sample per phase), main cylinder pressure (bar, one value per phase), proportional valve spool (%), ram/container position (mm, coarse per phase) | Demand flow (L/min) and demand pressure (bar) at `dt = 0.1 s` (0.2 s in OPC-UA server), ram position (mm), phase name, electricity tariff (EUR/MWh, hourly) |
| Output signals | `Diagnostics/ActiveStallFlag`, `Diagnostics/EstimatedExcessDelay` (s), anomaly type (VALVE_OVERLAP, MICRO_STALL, CREEPING_WEAR), HMM state, OEE and EUR recovery figures | `PumpSetpoints/StagingCode`, per-pump Active, DisplacementPct, PowerKW, FlowLPM; `EnergyAndTariffs/InstantaneousPower_KW`, `BaselinePower_KW`, `InstantaneousSavings_EUR_hr` |
| Config | `simulator/press_config.py` (dataclass `PressConfig`: 28 MN press, billet 850 mm x 228 mm, 95 kg, 8 phase timings); `dashboard/oee_calculator.py` (`PlantParameters`: 43 cycles/h, 7200 h/yr, 950 EUR/h, 450 EUR/t) | `simulator/press_demand.py` (`PhaseSpec` list, 7 phases); `data/raw_pump_curves.csv` (Bosch Rexroth A4VSO datasheet points); `data/electricity_prices_es.csv` |
| How it runs | `python -m simulator.press_state_machine`, `python -m opcua.opcua_server --speedup 20`, `streamlit run dashboard/dashboard_app.py` | `python -m opcua.opcua_server`, `streamlit run dashboard/app.py`, `python data/fetch_entsoe.py` (needs `ENTSOE_API_KEY`, else deterministic Spain fallback) |
| Tests | `tests/test_simulator.py`, `test_detector.py`, `test_opcua.py`, `test_end_to_end.py`, `smoke_test.py`, `validate_detector.py` (benchmark harness) | `tests/test_pumps.py`, `test_simulator.py`, `test_optimizer.py` |
| OPC-UA namespace | `urn:industrial:press:telemetry`, object path `Industrial_Plant/Press_01/{State,Telemetry,Diagnostics}` | `http://industrial.automation/hpeo/`, object path `HPEO_Press_HPU/{PressTelemetry,PumpSetpoints,EnergyAndTariffs}` |
| Endpoint | `opc.tcp://127.0.0.1:4840/freeopcua/server/` | `opc.tcp://127.0.0.1:4840/freeopcua/server/` |

## 2. Signals the three modules share or could share

| # | Signal | DCTO | HPEO | Skin-Flow Discard Optimizer use |
|---|---|---|---|---|
| 1 | Cycle phase | `State/CurrentPhase` (string) | `PressTelemetry/PressPhase` (string) | Gate the stroke window; timestamp shear |
| 2 | Cycle number | `State/CycleNumber` | `PressTelemetry/CycleId` | Join key for every per-cycle feature |
| 3 | Main cylinder pressure | `Telemetry/MainCylinderPressure` (one value per phase) | `PressTelemetry/DemandPressure_Bar` (10 Hz, demand not measured) | Ram force needs **cap and rod** pressure at 1 kHz; neither project provides it |
| 4 | Ram position | `Telemetry/ContainerPosition` (misnamed, carries ram position, coarse) | `ram_position_mm` in sim only (not published over OPC-UA) | Stroke coordinate x; needs 1 kHz |
| 5 | Phase durations | Per phase, per cycle | Per phase, per cycle (sim) | Shear travel time, dead-cycle coupling (Task 3.4) |
| 6 | Shear stroke duration | `shear_stroke` phase duration | `shear_stroke` phase duration | Cost of a thicker/thinner discard in seconds |
| 7 | Hydraulic flow | none | `DemandFlow_LPM`, per-pump `FlowLPM` | Ram speed cross-check |
| 8 | Pump power | none | per-pump `PowerKW`, `InstantaneousPower_KW` | Energy per cycle feature, ROI energy term |
| 9 | Electricity tariff | none | `GridTariff_EUR_MWh`, `data/electricity_prices_es.csv` | ROI energy term (reuse the loader) |
| 10 | Valve spool position | `Telemetry/ProportionalValveSpool` | none | Context feature only (read, never written) |

Signals the plan needs that **neither project has**: ram cap pressure and rod pressure (separate), container liner temperatures 1-4, billet temperature, oil temperature, supply-pressure minimum during stroke, butt shear position. These are defined fresh in `contracts/signals.yaml` with owner `skinflow`. See `docs/assumptions.md` A-02.

## 3. Naming conflicts and integration hazards

| # | Conflict | Detail | Resolution in this repo |
|---|---|---|---|
| C1 | Phase names differ | DCTO: `die_slide`, `billet_load`, `container_shift_close`, `rapid_advance`, `extrusion`. HPEO: `die_slide_shift`, `billet_load_and_seal`, `main_extrusion_stroke`, `inter_billet_dwell`. | Canonical enum in `contracts/signals.yaml` (`cycle_phase`), with alias maps from both projects |
| C2 | Phase count differs | DCTO 8 phases, HPEO 7 (no rapid advance or container close; adds dwell) | Canonical set is the union; aliases map onto it |
| C3 | Ram stroke length differs | DCTO billet 850 mm; HPEO sim ram advances 800 mm | Treated as billet-specific; `L0` is per billet, not a constant |
| C4 | Nominal timings differ | e.g. extrusion 65 s (DCTO) vs 38 s (HPEO); decompression 2.2 s vs 1.5 s | Our simulator takes its own values; coupling reads phase durations, never nominals |
| C5 | `ContainerPosition` carries ram position | DCTO writes `ram_position_mm` into a node called `ContainerPosition` | Contract separates `ram_position` and `container_position` |
| C6 | Same OPC-UA endpoint and port | Both bind `opc.tcp://127.0.0.1:4840/freeopcua/server/` | They cannot run together as-is. Our service subscribes; it never binds 4840. Mock mode uses 4842 |
| C7 | Same top-level package names | Both have `simulator/` and `opcua/` packages | Everything here lives under `skinflow_discard_optimizer/`; we never add the sibling repos to `sys.path` wholesale |
| C8 | Cycle id tag name | `CycleNumber` vs `CycleId` | Canonical `cycle_id` |
| C9 | numpy pin | DCTO pins `numpy<2.0`; this environment has numpy 2.4 | This repo targets numpy 2.x and does not import DCTO code at runtime |
| C10 | Pressure semantics | HPEO pressure is *demand*, DCTO pressure is a coarse *reading* | Contract field `kind: measured | demand` |
| C11 | HPEO writes setpoints | HPEO publishes pump staging setpoints | Our module is advisory only and writes only `Predictor.*` |

## 4. Reuse decisions

- **Tariff loader (HPEO `data/fetch_entsoe.py`, `mock_tariff_fallback.py`):** reused by *reading its CSV output format* (`timestamp_utc, price_eur_per_mwh, price_eur_per_kwh`) through `skinflow_discard_optimizer/roi/tariff.py`. We do not import the module because it does a bare `from mock_tariff_fallback import ...` that only works from inside HPEO's `data/` directory.
- **Viscosity model:** HPEO has **no oil temperature or viscosity model**. Task 1.3's supply-pressure sag uses our own placeholder Walther-type model (assumption A-05).
- **Dead-cycle timing:** Task 3.4 reads shear-stroke timing from the DCTO `PhaseTiming` values (nominal 2.5 s, std 0.10 s), copied into `config/coupling.yaml` with a `SOURCE:` pointing to `DCTO/simulator/press_config.py`.
