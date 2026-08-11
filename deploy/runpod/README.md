# RunPod deployment

One-time setup builds a reusable template; after that, "rent pod → wait ~90 s
→ SSH in with the repo synced and the venv ready".

## One-time setup

### 1. GitHub token (repo read access for the pod)

GitHub → Settings → Developer settings → Fine-grained personal access tokens →
Generate new token. Repository access: **only `pred_el_prices`**. Permissions:
**Contents: Read-only**. Copy the token.

### 2. RunPod secret

RunPod console → Settings → Secrets → Create Secret:
name `github_pat`, value = the token.

### 3. SSH key

RunPod console → Settings → SSH Public Keys → paste your public key
(`~/.ssh/id_ed25519.pub`). The official images install it automatically.

### 4. Template

RunPod console → Templates → New Template:

| field | value |
|---|---|
| Container Image | `runpod/pytorch:1.1.0-rc.146-cu1281-torch291-ubuntu2404` (newest torch/CUDA pairing as of 2026-07; the image's own python/torch go unused — uv provisions the env from the lockfile) |
| Container Start Command | see below |
| Container Disk | 20 GB |
| Volume | 30 GB mounted at `/workspace` (survives pod stop/restart) |
| Env var `GITHUB_PAT` | `{{ RUNPOD_SECRET_github_pat }}` |
| Expose SSH | keep TCP port 22 exposed (default in official images) |

Container Start Command (one line):

```
bash -c 'curl -sf -H "Authorization: token $GITHUB_PAT" https://raw.githubusercontent.com/baakflo/pred_el_prices/main/deploy/runpod/bootstrap.sh | bash; /start.sh'
```

The bootstrap clones/pulls the repo into `/workspace/pred_el_prices` and runs
`uv sync`; `/start.sh` is the image's own entrypoint that starts sshd and
keeps the container alive. Because `/workspace` is the persistent volume, a
stopped-and-restarted pod re-boots in seconds (git pull + cached venv).

## Daily use

Rent a pod from the template, then from any machine:

```
ssh root@<pod-ip> -p <mapped-port>          # connection details on the pod page
cd /workspace/pred_el_prices
uv run pep run <experiment>                  # experiment registry (see configs/experiments/)
```

Push the local market-data cache to a fresh pod (from the laptop):

```
tar -czf cache.tgz -C data cache
scp -P <port> cache.tgz root@<ip>:/workspace/pred_el_prices/data/
ssh root@<ip> -p <port> "cd /workspace/pred_el_prices/data && tar -xzf cache.tgz && rm cache.tgz"
```

Fetch results:

```
scp -P <port> -r root@<ip>:/workspace/pred_el_prices/runs/<job_id> ./runs/
```

## ECMWF ENS backfill

`backfill_ecmwf.sh` shards run dates round-robin across N parallel workers
(each date is one idempotent `pep backfill-ecmwf` call; safe to re-run, only
missing dates are fetched). S3 needs no credentials. Start with 4 workers and
watch the logs for `503 Slow Down` pile-ups before scaling up:

```
cd /workspace/pred_el_prices
nohup bash deploy/runpod/backfill_ecmwf.sh 2023-01-18 2026-08-10 4 > logs/backfill.log 2>&1 &
tail -f logs/backfill_w0.log
```

Fetch the archive home (from the laptop):

```
scp -P <port> -r root@<ip>:/workspace/pred_el_prices/data/archive/weather/ecmwf-ens ./data/archive/weather/
```

## Notes

- Secrets stay in RunPod; nothing sensitive is in the repo or the template
  besides the secret *reference*.
- The ENTSO-E key is not needed on pods (they train from the shipped cache);
  if a pod must fetch data, add `ENTSOE_API_KEY` the same way as the PAT.
- Untested-by-construction until the first real pod boot; if the start
  command misbehaves, the pod's container logs (pod page → Logs) show the
  bootstrap output line by line.
