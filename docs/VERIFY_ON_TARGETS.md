# Verify on Windows, ARM, Docker and systemd

These paths were **built but not run** by the author of this tooling (no Windows, ARM64 or Docker daemon was available).
Run the commands below on each target and keep the output. A path counts as verified only when its last command
prints `PREFLIGHT PASSED` (or the service shows `healthy`).

## Windows 10/11 (PowerShell)

Option A - venv:
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
skdo probe
$env:SKDO_TIER="edge"; skdo bootstrap          # about 5 minutes
skdo preflight
pytest tests/test_deploy.py tests/test_bundle.py
```
Option B - USB stick built with `--embed-python` (nothing installed): double-click `START-WINDOWS.bat`, or in `cmd`:
```bat
D:\skdo\START-WINDOWS.bat --no-run
```
Things most likely to need fixing on Windows: the `.bat` quoting, the embedded `python312._pth` paths, long paths on the stick,
and process pools under `spawn`.

## ARM (Raspberry Pi 4/5 or any aarch64 Linux, 64-bit OS)

```bash
python3 --version                  # 3.11+
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # needs aarch64 wheels: numpy/scipy/pandas/pyarrow publish them for cp312/cp313
skdo probe                         # expect tier=edge on a Pi
SKDO_TIER=edge skdo bootstrap && skdo preflight
```
USB stick built with `--target cp312-linux_aarch64`: `sh start.sh --no-run`.
No ARM hardware? Emulate (slow): `docker run --privileged --rm tonistiigi/binfmt --install arm64` then build with `--platform linux/arm64`.

## Docker
```bash
docker build -t skdo .
docker run --rm skdo -- --mock --mock-cycles 20                       # demo, no press
docker run --rm -v skdo-art:/app/artifacts -v skdo-data:/app/data skdo --no-run   # builds models, runs preflight, exits 0 if it passed
docker compose up -d && docker compose ps                             # health becomes "healthy" after models are built
docker compose logs -f skdo
```
GPU: `docker build -t skdo:gpu --build-arg EXTRAS=gpu .` then `docker run --rm --gpus all skdo:gpu --no-run`.
First start builds models inside the container volume; the health check allows 1 hour for that.

## systemd (a real Linux host with systemd, as root)
```bash
sudo deploy/install.sh                       # --gpu optional
sudoedit /etc/skdo/skdo.env                  # set SKDO_OPCUA_ENDPOINT (and press id / namespace)
sudo systemctl enable --now skdo
systemctl status skdo --no-pager
journalctl -fu skdo                          # watch bootstrap, preflight, then live cycles
sudo systemctl restart skdo && journalctl -u skdo -n 30 --no-pager
```
Check the service user can write: `sudo -u skdo test -w /opt/skdo/app/artifacts && echo writable`.
To test without a press, run `skdo simulate-press` on the same host and set `SKDO_OPCUA_ENDPOINT=opc.tcp://127.0.0.1:4842/press`.

## Anything, anywhere
```bash
skdo doctor          # install checks and smoke test
skdo preflight       # the gate
pytest -m "not slow" --ignore=tests/test_service.py --ignore=tests/test_fault_id.py
pytest -m slow tests/test_opcua_live.py      # needs built models, about 1-2 minutes
```
