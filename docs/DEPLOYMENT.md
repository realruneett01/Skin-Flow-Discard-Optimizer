# Deployment

```
skdo up        probe -> configure -> bootstrap -> PREFLIGHT GATE -> run
```

`skdo up` reads the host, sizes a model-build profile to it, configures the press's signal contract, builds
models if they are missing, runs the preflight gate, and only then starts. If the gate fails, nothing starts.

## Status of each path (be honest about what has been run)

| Path | Built | Run end to end |
|---|---|---|
| Linux x86_64 (Python 3.11, 3.12, 3.13): install, test suite | yes | **yes**, ~4,400 tests per Python version in clean virtual environments |
| Linux x86_64, USB bundle (libraries loaded from the stick, tamper check) | yes | **yes** |
| `skdo up` live against the **simulated** press, `Predictor.*` read back over OPC-UA by a client | yes | **yes** |
| Encrypted + authenticated OPC-UA, link-loss recovery | yes | **yes**, against the simulated press |
| First start from nothing (fresh venv, same steps as the Dockerfile layers): build models, preflight, run | yes | **yes** (8 min at the `standard` tier on 4 cores) |
| OPC-UA against a **real** PLC | yes | **no** |
| Windows, Linux ARM64, macOS bundles | yes (wheels resolve for cp312 win_amd64 / linux_aarch64 / macos_arm64) | **no**, see `docs/VERIFY_ON_TARGETS.md` |
| Docker image build and run, docker compose | yes | **no** (no Docker daemon was available; only the Dockerfile's install steps were replayed in a venv) |
| systemd unit | yes | syntax checked with `systemd-analyze verify` (the only complaint was that the binary does not exist on the build machine); **never started** |
| CUDA GPU path | yes | **no** (no GPU; the CPU-torch version of the GPU engine is tested) |
| GitHub Actions workflow | yes | **no** (YAML parses; it has never run) |

## USB stick

Build once on any machine with internet:

```
skdo bootstrap                                   # build models on a strong machine first (SKDO_TIER=full recommended)
skdo configure --press-id Press07 --namespace 3 --endpoint opc.tcp://192.168.0.10:4840
skdo bundle /media/usb/skdo \
    --target cp312-win_amd64 --target cp312-linux_x86_64 --target cp312-linux_aarch64 \
    --embed-python python-3.12.x-embed-amd64.zip      # Windows machines then need nothing installed
```

Get the embeddable zip from python.org (Python 3.12, "Windows embeddable package (64-bit)"). Each target adds
roughly 250-300 MB of unpacked libraries; prebuilt models and data are copied from `artifacts/` and `data/`.

On the target machine, plug in the stick and start the launcher: `START-WINDOWS.bat`, `sh start.sh`, or
`start-macos.command`. Operating systems do not auto-run programs from USB drives, and this software does not
try to make them. The launcher:

1. selects the libraries matching this machine's OS, CPU and Python (nothing is installed; state stays on the stick);
2. **verifies the stick has not been altered** against `MANIFEST.sha256` (code, config and libraries are hashed; models and logs are not);
3. keeps the tier the models were built with, so prebuilt models are used as they are;
4. runs the **preflight gate** on *this* machine (the result is bound to the machine, the config, the contract and the models);
5. starts the live service if an OPC-UA endpoint is configured.

## The preflight gate (`skdo preflight`)

| Check | Fails the gate when |
|---|---|
| python, contract | Python < 3.11, contract invalid or contains a setpoint-like tag |
| state dir, artifacts | state directory not writable, models missing |
| decision cycle | a real cycle does not complete inside the [12, 60] mm bounds |
| OPC-UA loopback | the live path over OPC-UA does not give the same cut as direct replay of the same cycle (a simulated press is served on local ports) |
| Predictor nodes read-only | a client could write the advisory tags |
| required tags deliver data | a configured press endpoint does not deliver all 16 tags during the listening window |
| values in range, monotonic timestamps | tags are out of contract range or timestamps go backwards |

Warnings (do not fail): latency over the 200 ms budget, models built with a different config, sample rates
below half the contract rate, ram signals constant (run commissioning while a billet is being extruded),
and PLACEHOLDER parameters.

### What "calibrates itself" means here

- **Signal-level commissioning: yes.** Presence, range, rate, timestamps and variation of every required tag are
  measured on the real press and written to `artifacts/commissioning.json`.
- **Host-level sizing: yes.** Model size and workers follow the machine.
- **Process-model calibration: no.** Fitting the physics and defect model to your press needs matched force curves
  and discard-quality labels (README, "Pilot programme"). Nothing here can learn that from the tags alone.

## Other install routes

| Where | How |
|---|---|
| Docker (CPU) | `cp deploy/skdo.env.example deploy/skdo.env`, edit, `docker compose up -d` |
| Docker (NVIDIA GPU) | `docker compose --profile gpu up -d skdo-gpu` |
| Bare Linux + systemd | `sudo deploy/install.sh`, edit `/etc/skdo/skdo.env`, `sudo systemctl enable --now skdo` |
| Developer | `pip install -e ".[dev]" && skdo up -- --mock --mock-cycles 50` |
| No press, test the whole path | `skdo simulate-press` in one terminal, `skdo up --endpoint opc.tcp://127.0.0.1:4842/press` in another |

## What gets auto-configured

**From the host** (`skdo probe`): tier from usable cores and RAM (container limits respected): `edge` (<=2 cores or <4 GB),
`standard`, `full` (>=8 cores and >=16 GB); GPU use when CUDA torch is present. Existing models keep their tier.

**From the press** (`site/site.yaml`, or `SKDO_PRESS_ID`, `SKDO_OPCUA_NAMESPACE`, `SKDO_OPCUA_ENDPOINT` on first start):
`skdo configure` rewrites `contracts/signals.yaml` for that press and validates it before saving.
`Predictor.*` outputs cannot be overridden.

**From the config**: `skdo bootstrap` fingerprints `config/*.yaml` and rebuilds only the affected stages.

Overrides: `SKDO_TIER`, `SKDO_WORKERS`, `SKDO_DEVICE`, `SKDO_DATASET_SCALE`, `SKDO_LATENCY_BUDGET_MS`,
`SKDO_OPCUA_USER`/`SKDO_OPCUA_PASSWORD` (client username auth).

## Security

- The service hosts a **read-only** `Predictor.*` OPC-UA server, by default on `opc.tcp://127.0.0.1:4841/skdo`.
  It refuses to bind a routable address unless it is encrypted: create a certificate with `skdo gen-cert`, then set
  `SKDO_SERVE_CERT` and `SKDO_SERVE_KEY` (Basic256Sha256 SignAndEncrypt). Add `SKDO_SERVE_USER` and
  `SKDO_SERVE_PASSWORD` to require a login (anonymous access is then refused). `SKDO_ALLOW_INSECURE=1` overrides, for an isolated network only.
- The connection **to the press** is encrypted when `SKDO_OPCUA_CERT`, `SKDO_OPCUA_KEY` and `SKDO_OPCUA_SERVER_CERT`
  are set (trust the certificate on the press server too); otherwise a warning is logged. `SKDO_OPCUA_USER`/`SKDO_OPCUA_PASSWORD` add a login.
- Covered by tests (`tests/test_opcua_security.py`): encrypted and authenticated round trip, wrong password rejected,
  anonymous rejected, unencrypted client rejected, clients cannot write.
- Not covered: certificate expiry and rotation, trust-list management, and interoperability with a specific PLC vendor's OPC-UA server.

## Behaviour when things go wrong

- A cycle with corrupt telemetry, an unsupported alloy/temperature, a stroke that stops short of the decision region,
  a bad billet length, or missing models gets the **static 40 mm cut**, with the reasons logged and published as
  `Predictor.CutConfidence = fallback`. It does not reach the drift monitor.
- An unexpected exception in the decision engine is logged with its traceback and handled the same way (`ENGINE_ERROR`).
  One bad cycle never stops the service.
- A press server that sends wrong types or garbage has those values counted (`LiveRunner.bad_values`) and dropped.
- If no data arrives for 30 s and the link is dead, the service drops the half-assembled cycle and reconnects.
- `Predictor.CutIntervalLowMm/HighMm` are the calibrated conformal interval on the **critical thickness**, not on the cut. The
  cost-optimal cut normally sits above that interval by design.

## Limits you should know about

- **Everything is built from a digital twin with PLACEHOLDER parameters** until calibrated on real force curves.
  A passing preflight means the software is installed and wired correctly, not that its cuts are right for your press.
- **Only a simulated press has been used** to test the OPC-UA path. A real PLC's sampling limits, quality codes and
  vendor quirks are untested.
- **1 kHz delivery over OPC-UA depends on the press server.** Commissioning reports the observed rates.
- **Smaller tiers give less-validated models.** Coverage and backtest numbers come from the full tier. For real presses,
  build once on a `full` host and ship the models (USB bundle).
- **Latency:** on a 4-vCPU Linux host the full decision path measured a median of about 140 ms and a p99 of about
  195 ms against the 200 ms budget (after the optimisations in this repository). That is a margin of a few percent: a slower CPU
  will exceed it. The budget only produces a warning; the cut is still published. The first cycle after start is slower.
- The bundle verifies integrity, it does not authenticate the author (no signature). Treat the stick as you treat any
  executable media.
