"""Build a self-contained folder to copy onto a USB stick and run on a machine with nothing installed.

    skdo bundle /media/usb/skdo --target cp312-win_amd64 --target cp312-linux_x86_64 --embed-python py312.zip

Layout::

    START-WINDOWS.bat | start.sh | start-macos.command    launchers (verify -> up)
    app/       the package, config/, contracts/, LICENSE, prebuilt artifacts/ and data/ (state lives here)
    lib/<cpXYZ-platform>/    third-party wheels, already unpacked: nothing is installed on the host
    runtime/win_amd64/       optional embeddable Python for Windows
    MANIFEST.sha256          hashes of code and libraries; the launcher refuses a bundle that fails it

Nothing is written outside the folder. Prebuilt models are only valid for the config they were built with
(`skdo bootstrap` rebuilds if the config differs). Operating systems do not (and should not) auto-run
programs from a USB stick: the operator double-clicks the launcher.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# target platform -> pip --platform tags
PLATFORMS = {
    "win_amd64": ["win_amd64"],
    "linux_x86_64": ["manylinux2014_x86_64", "manylinux_2_28_x86_64", "manylinux_2_17_x86_64"],
    "linux_aarch64": ["manylinux2014_aarch64", "manylinux_2_28_aarch64", "manylinux_2_17_aarch64"],
    "macos_arm64": ["macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_14_0_arm64"],
    "macos_x86_64": ["macosx_10_13_x86_64", "macosx_11_0_x86_64", "macosx_12_0_x86_64"],
}
MUTABLE = ("app/artifacts", "app/data", "app/site", "app/reports")   # change at run time: not hashed


def parse_target(t: str) -> tuple[str, str, str]:
    """'cp312-win_amd64' -> ('cp312', '3.12', 'win_amd64')."""
    tag, _, plat = t.partition("-")
    if not (tag.startswith("cp") and tag[2:].isdigit() and plat in PLATFORMS):
        raise ValueError(f"bad target {t!r}; use cpXYZ-<{'|'.join(PLATFORMS)}>")
    return tag, f"{tag[2]}.{tag[3:]}", plat


def runtime_dependencies() -> list[str]:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    return list(deps)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".pytest_cache"),
                    dirs_exist_ok=True)


def _install_libs(target: str, dest: Path, log) -> None:
    tag, pyver, plat = parse_target(target)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(dest), "--only-binary=:all:",
           "--python-version", pyver, "--implementation", "cp", "--abi", tag, "--no-compile"]
    for p in PLATFORMS[plat]:
        cmd += ["--platform", p]
    cmd += runtime_dependencies()
    log(f"[bundle] {target}: downloading and unpacking wheels")
    subprocess.run(cmd, check=True)


def _write_launchers(out: Path, targets: list[str]) -> None:
    (out / "start.sh").write_text(_UNIX_LAUNCHER, newline="\n")
    (out / "start-macos.command").write_text('#!/bin/sh\ncd "$(dirname "$0")" && exec sh ./start.sh "$@"\n', newline="\n")
    (out / "START-WINDOWS.bat").write_text(_WIN_LAUNCHER.replace("\n", "\r\n"), newline="")
    (out / "README-USB.txt").write_text(_USB_README.format(targets=", ".join(targets)), encoding="utf-8")


def _embed_python(zip_path: Path, out: Path, targets: list[str]) -> None:
    win = [t for t in targets if t.endswith("win_amd64")]
    if not win:
        raise ValueError("--embed-python needs a cpXYZ-win_amd64 target")
    dest = out / "runtime" / "win_amd64"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    pth = next(dest.glob("python*._pth"))
    lines = [ln for ln in pth.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    # An embeddable Python ignores PYTHONPATH when a ._pth file exists, so the paths go here.
    lines += [f"..\\..\\app", f"..\\..\\lib\\{win[0]}", "import site"]
    pth.write_text("\n".join(dict.fromkeys(lines)) + "\n")


def manifest_entries(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if (p.is_file() and rel != "MANIFEST.sha256" and not rel.startswith(MUTABLE)
                and "__pycache__" not in p.parts):      # bytecode caches are not code
            files.append(p)
    return files


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path) -> int:
    lines = [f"{_sha(p)}  {p.relative_to(root).as_posix()}" for p in manifest_entries(root)]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_manifest(root: Path) -> list[str]:
    """Problems found (empty = intact): modified, missing or unexpected files."""
    mf = root / "MANIFEST.sha256"
    if not mf.exists():
        return ["MANIFEST.sha256 is missing"]
    expected = {}
    for ln in mf.read_text(encoding="utf-8").splitlines():
        digest, _, rel = ln.partition("  ")
        expected[rel] = digest
    problems = []
    seen = set()
    for p in manifest_entries(root):
        rel = p.relative_to(root).as_posix()
        seen.add(rel)
        if rel not in expected:
            problems.append(f"unexpected file: {rel}")
        elif _sha(p) != expected[rel]:
            problems.append(f"modified: {rel}")
    problems += [f"missing: {r}" for r in expected if r not in seen]
    return problems


def build_bundle(out: Path, targets: list[str], with_artifacts: bool = True, embed_python: str | None = None,
                 log=print) -> None:
    for t in targets:
        parse_target(t)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    app = out / "app"
    for d in ("skinflow_discard_optimizer", "config", "contracts"):
        _copy_tree(ROOT / d, app / d)
    for f in ("pyproject.toml", "LICENSE"):
        shutil.copy2(ROOT / f, app / f)
    if (ROOT / "site").exists():
        _copy_tree(ROOT / "site", app / "site")
    if with_artifacts:
        for d in ("artifacts", "data"):
            if (ROOT / d).exists():
                _copy_tree(ROOT / d, app / d)
                for stale in ("preflight.json", "commissioning.json"):   # results belong to one machine
                    (app / d / stale).unlink(missing_ok=True)
                log(f"[bundle] copied prebuilt {d}/ (models are valid only for this config)")
    for t in targets:
        _install_libs(t, out / "lib" / t, log)
    if embed_python:
        _embed_python(Path(embed_python), out, targets)
    _write_launchers(out, targets)
    n = write_manifest(out)
    log(f"[bundle] done: {out} ({n} files hashed). Copy this folder to the USB stick.")


_UNIX_LAUNCHER = r"""#!/bin/sh
# Plug-and-run launcher (Linux / macOS). Usage: sh start.sh [skdo up options]
HERE="$(cd "$(dirname "$0")" && pwd)"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64|Linux-amd64)   PLAT=linux_x86_64;;
  Linux-aarch64|Linux-arm64)  PLAT=linux_aarch64;;
  Darwin-arm64)               PLAT=macos_arm64;;
  Darwin-x86_64)              PLAT=macos_x86_64;;
  *) echo "Unsupported machine: $(uname -s) $(uname -m)"; exit 1;;
esac
PY=""; TAG=""
for c in python3.13 python3.12 python3.11 python3 python; do
  command -v "$c" >/dev/null 2>&1 || continue
  t="$("$c" -c 'import sys; print("cp%d%d" % sys.version_info[:2])' 2>/dev/null)" || continue
  if [ -d "$HERE/lib/$t-$PLAT" ]; then PY="$c"; TAG="$t"; break; fi
done
if [ -z "$PY" ]; then
  echo "No Python on this machine matches the libraries on this stick:"
  ls "$HERE/lib"; echo "Install one of those Python versions, or rebuild the bundle for yours."; exit 1
fi
export PYTHONPATH="$HERE/app:$HERE/lib/$TAG-$PLAT" PYTHONDONTWRITEBYTECODE=1
"$PY" -m skinflow_discard_optimizer.deploy.cli verify-bundle "$HERE" || exit 1
exec "$PY" -m skinflow_discard_optimizer.deploy.cli up "$@"
"""

_WIN_LAUNCHER = r"""@echo off
rem Plug-and-run launcher (Windows). Double-click, or run from cmd: START-WINDOWS.bat [skdo up options]
setlocal
set "HERE=%~dp0"
if exist "%HERE%runtime\win_amd64\python.exe" (set "PY=%HERE%runtime\win_amd64\python.exe") else (set "PY=python")
set "TAG="
for /f "usebackq delims=" %%t in (`"%PY%" -c "import sys; print('cp%%d%%d-win_amd64' %% sys.version_info[:2])"`) do set "TAG=%%t"
if not defined TAG (
  echo Python was not found. Use a bundle built with --embed-python, or install Python 3.11+ from python.org.
  pause & exit /b 1
)
if not exist "%HERE%lib\%TAG%" (
  echo This stick has no libraries for %TAG%. Available:
  dir /b "%HERE%lib"
  pause & exit /b 1
)
set "PYTHONPATH=%HERE%app;%HERE%lib\%TAG%"
set "PYTHONDONTWRITEBYTECODE=1"
"%PY%" -m skinflow_discard_optimizer.deploy.cli verify-bundle "%HERE%" || (pause & exit /b 1)
"%PY%" -m skinflow_discard_optimizer.deploy.cli up %*
pause
"""

_USB_README = """Skin-Flow Discard Optimizer - plug-and-run stick
=================================================
Built for: {targets}

1. Plug the stick into the computer that is on the press network.
2. Start the launcher (an OS will not auto-run it, by design):
     Windows : double-click START-WINDOWS.bat
     Linux   : sh start.sh
     macOS   : double-click start-macos.command   (or: sh start.sh)
3. The launcher verifies this stick has not been altered, picks settings for this computer,
   configures from app/site/site.yaml, runs the PREFLIGHT gate, and only then starts.
   First-time press set-up: run once with options, for example
     sh start.sh --endpoint opc.tcp://192.168.0.10:4840
   (the press address is remembered in app/site/site.yaml after `skdo configure`).

The service is ADVISORY ONLY. It reads press tags and publishes read-only Predictor.* tags on
opc.tcp://<this computer>:4841/skdo. It never writes to the press.

If preflight FAILS the service does not start. Read the FAIL lines.
Nothing is installed on the computer; all state is under app/ on this stick.
Use of this software is governed by LICENSE (all rights reserved, proprietary).
"""
