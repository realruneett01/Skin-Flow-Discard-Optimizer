# Deployment

One command per machine. The deploy layer reads the host, sizes a model-build profile to it,
configures the press's signal contract, builds the models on first start, proves the result with a
smoke test, then runs the service.

```
skdo up        probe -> configure -> bootstrap -> smoke test -> run
```

## Options

| Where | How |
|---|---|
| Docker (CPU) | `cp deploy/skdo.env.example deploy/skdo.env`, edit, `docker compose up -d` |
| Docker (NVIDIA GPU) | `docker compose --profile gpu up -d skdo-gpu` (needs nvidia-container-toolkit) |
| Bare Linux + systemd | `sudo deploy/install.sh` (adds `--gpu` automatically when `nvidia-smi` sees a GPU) |
| Developer machine | `pip install -e ".[dev]" && skdo up -- --mock --mock-cycles 50` |

## What gets auto-configured

**From the host** (`skdo probe` shows the result):

| Detected | Effect |
|---|---|
| Usable cores (honours cgroup CPU limits and affinity), RAM (honours cgroup limit) | Picks a tier: `edge` (≤2 cores or <4 GB), `standard`, `full` (≥8 cores and ≥16 GB). The tier sets simulated dataset size, fit sample sizes and worker count. |
| NVIDIA GPU and CUDA-enabled torch | Decision fitting and batched inference run on `cuda`. Edge tier stays on CPU. |
| Container vs bare metal | Reported only. |

**From the press** (`site/site.yaml`, or `SKDO_PRESS_ID`, `SKDO_OPCUA_NAMESPACE`, `SKDO_OPCUA_ENDPOINT` on first start):
`skdo configure` rewrites `contracts/signals.yaml` for that press (tag prefix, OPC-UA namespace index,
per-signal tag overrides) into `site/signals.yaml` and validates it with the contract schema before writing
anything. A bad override fails at configure time. `Predictor.*` output tags cannot be overridden, so the
advisory-only guarantee holds on every site.

```
skdo configure --press-id Press07 --namespace 3 --endpoint opc.tcp://10.0.0.5:4840 \
               --tag ram_cap_pressure=PLC.Main.CapP
```

**From the config**: `skdo bootstrap` fingerprints `config/*.yaml` and the scenarios. Change a press
parameter and only the affected downstream stages rebuild. `skdo bootstrap --status` shows
`done` / `stale` / `missing` per stage. Workers and device are left out of the fingerprint, so models built on a big
host are valid on a small one when you pin `SKDO_TIER=full` there.

Overrides: `SKDO_TIER`, `SKDO_WORKERS`, `SKDO_DEVICE`, `SKDO_DATASET_SCALE`, `SKDO_LATENCY_BUDGET_MS`.

## Check an install

`skdo doctor` checks Python and dependencies, the signal contract (and that it has no setpoint tags), config
provenance (warns how many parameters are still PLACEHOLDER), writable artifact directory, model artifacts
present and current, and runs a 3-cycle smoke test through the real service. Exit code 1 on any failure.
The container health check runs `skdo doctor --quick`.

## Limits you should know about

- **No live OPC-UA client exists in this repository.** `service/run.py` runs `--mock` (digital twin) or
  `--replay <parquet>`. The README's "live OPC-UA mode" is not implemented. `site.yaml` stores the endpoint
  and the contract carries the tags, but something has to subscribe to them and feed `CycleAssembler`. That
  adapter is the main thing missing for a real press. The default container command therefore runs a short
  mock stream.
- **Smaller tiers produce less-validated models.** The 90 % interval coverage and the backtest numbers were produced
  at full scale. `edge` and `standard` trade calibration data for build time (the conformal calibration set is
  a few hundred cycles instead of ~1,500). For real presses, build once on a `full` host and copy `artifacts/` over.
- **All models are built from the digital twin with PLACEHOLDER parameters** until calibrated on real force curves.
  Treat any output as a demonstration of the pipeline, not of savings.
- The bootstrap time depends on the tier. Measure it on your hardware with `skdo bootstrap`.
