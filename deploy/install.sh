#!/usr/bin/env bash
# Install on a bare Linux host (run from a checkout, as root):  sudo deploy/install.sh [--gpu] [--no-systemd] [--prefix DIR]
set -euo pipefail

PREFIX=/opt/skdo; EXTRAS=""; SYSTEMD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) EXTRAS="[gpu]";;
    --no-systemd) SYSTEMD=0;;
    --prefix) PREFIX="$2"; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
  shift
done

SRC="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-}"
for c in python3.13 python3.12 python3.11 python3; do
  [[ -n "$PY" ]] && break
  command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' && PY="$c"
done
[[ -n "$PY" ]] || { echo "Python >= 3.11 is required" >&2; exit 1; }

# Auto-enable the GPU extra when an NVIDIA GPU is visible.
if [[ -z "$EXTRAS" ]] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  EXTRAS="[gpu]"; echo "NVIDIA GPU detected: installing the gpu extra"
fi

mkdir -p "$PREFIX"
# The package resolves config/, contracts/ and artifacts/ relative to its source tree, so install it in place.
mkdir -p "$PREFIX/app"
cp -a "$SRC"/pyproject.toml "$SRC"/LICENSE "$SRC"/skinflow_discard_optimizer "$SRC"/config "$SRC"/contracts "$PREFIX/app/"
"$PY" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip
(cd "$PREFIX/app" && "$PREFIX/venv/bin/pip" install --quiet -e ".${EXTRAS}")

if [[ $SYSTEMD -eq 1 ]]; then
  id skdo >/dev/null 2>&1 || useradd --system --home "$PREFIX" --shell /usr/sbin/nologin skdo
  chown -R skdo:skdo "$PREFIX"
  mkdir -p /etc/skdo
  [[ -f /etc/skdo/skdo.env ]] || cp "$SRC/deploy/skdo.env.example" /etc/skdo/skdo.env
  sed "s#/opt/skdo#$PREFIX#g" "$SRC/deploy/skdo.service" > /etc/systemd/system/skdo.service
  systemctl daemon-reload
  echo "Edit /etc/skdo/skdo.env, then:  systemctl enable --now skdo   (first start builds models; follow with journalctl -fu skdo)"
else
  echo "Installed. Run:  $PREFIX/venv/bin/skdo up"
fi
"$PREFIX/venv/bin/skdo" probe | head -30
