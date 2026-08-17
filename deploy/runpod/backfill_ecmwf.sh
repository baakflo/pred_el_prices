#!/usr/bin/env bash
# Parallel ECMWF ENS backfill: interleaves run dates across N worker loops.
# Each worker calls the idempotent per-date archiver (existing Parquets are
# skipped), so the script can be re-run after any failure and only missing
# dates are fetched. Worker logs land in logs/backfill_w<k>.log.
#
# Usage: bash deploy/runpod/backfill_ecmwf.sh START END [N_WORKERS=4] [RUN_HOUR=0]
# e.g.:  nohup bash deploy/runpod/backfill_ecmwf.sh 2023-02-01 2023-02-28 4 12 \
#          > logs/backfill.log 2>&1 &
set -euo pipefail
START=$1
END=$2
WORKERS=${3:-4}
RUN_HOUR=${4:-0}
cd "$(dirname "$0")/../.."
mkdir -p logs

mapfile -t ALL < <(uv run python -c "
from datetime import date, timedelta
d, end = date.fromisoformat('$START'), date.fromisoformat('$END')
while d <= end:
    print(d)
    d += timedelta(days=1)
")

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
