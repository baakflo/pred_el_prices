#!/usr/bin/env bash
# Parallel ECMWF ENS backfill over an explicit date list (one ISO date per
# line): the multi-pod shard variant of backfill_ecmwf.sh. Each worker calls
# the idempotent per-date archiver, so re-runs only fetch missing dates.
# Worker logs land in logs/backfill_w<k>.log.
#
# Usage: bash deploy/runpod/backfill_dates.sh DATES_FILE [N_WORKERS=4] [RUN_HOUR=12]
set -euo pipefail
LIST=$(realpath "$1")
WORKERS=${2:-4}
RUN_HOUR=${3:-12}
cd "$(dirname "$0")/../.."
mkdir -p logs

mapfile -t ALL < <(grep -v '^\s*$' "$LIST")

for ((k = 0; k < WORKERS; k++)); do
  (
    for ((i = k; i < ${#ALL[@]}; i += WORKERS)); do
      d=${ALL[i]}
      uv run pep backfill-ecmwf --start "$d" --end "$d" --run-hour "$RUN_HOUR" || echo "FAIL $d"
    done
    echo "worker $k done"
  ) >"logs/backfill_w$k.log" 2>&1 &
done
wait
echo "all $WORKERS workers finished; archive under data/archive/weather/ecmwf-ens/"
