# Laptop-side incremental pull of the pod's ECMWF archive (run hourly via
# Task Scheduler while a multi-day backfill runs on a preemptible pod).
# Only fetches Parquets missing locally; the run log lands next to the data.
#
# Register (adjust host/port):
#   schtasks /Create /SC HOURLY /TN pep-pull-ecmwf /F /TR "powershell -NoProfile
#     -ExecutionPolicy Bypass -File <repo>\deploy\runpod\pull_archive.ps1
#     -SshHost <ip> -Port <port>"
param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot)
$remoteRoot = "/workspace/pred_el_prices/data/archive/weather/ecmwf-ens"
$localRoot = Join-Path $repo "data\archive\weather\ecmwf-ens"
$log = Join-Path $repo "data\archive\weather\pull_archive.log"

$stamp = (Get-Date).ToUniversalTime().ToString("s")
try {
    $remote = ssh -o ConnectTimeout=15 -p $Port -i $Key "root@$SshHost" find $remoteRoot -type f |
        Where-Object { $_ -like "*.parquet" }
    $pulled = 0
    foreach ($r in $remote) {
        $rel = $r.Substring($remoteRoot.Length + 1) -replace "/", "\"
        $dest = Join-Path $localRoot $rel
        if (-not (Test-Path $dest)) {
            New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
            scp -q -o ConnectTimeout=15 -P $Port -i $Key "root@${SshHost}:$r" $dest
            $pulled++
        }
    }
    Add-Content $log "$stamp OK remote=$($remote.Count) pulled=$pulled"
} catch {
    Add-Content $log "$stamp ERROR $($_.Exception.Message)"
    exit 1
}
