#!/usr/bin/env bash
# RunPod bootstrap: idempotent setup of pred_el_prices on a fresh pod.
# Expects GITHUB_PAT in the environment (injected via a RunPod secret).
set -euo pipefail

REPO_DIR=/workspace/pred_el_prices

if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "https://${GITHUB_PAT}@github.com/baakflo/pred_el_prices.git" "$REPO_DIR"
fi

# GRIB decoding: the Linux eccodes wheel has no bundled binary (same as CI)
dpkg -s libeccodes-dev >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq libeccodes-dev; }

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
uv sync --extra dev

echo "bootstrap done: $(git rev-parse --short HEAD)"
