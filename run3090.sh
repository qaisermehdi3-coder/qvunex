#!/usr/bin/env bash
# RTX 3090 — Qaiser's OWN second 3090, not client work. Prepared 15 Oct 2026.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run3090.sh | bash
#
# Why: on Seth's job (19 Sept) one RTX 3090 came out 0.32-0.88x the L40S on
# every CUDA-graph config, where a bandwidth ratio predicts 1.08x. One card on
# one host cannot say whether that is the 3090 or that unit. This is the same
# script and flags on a different 3090, on Qaiser's own account, so the
# numbers are his to publish.
#
# Host facts the container cannot see reliably (vCPUs allocated, RAM, disk,
# machine and host id, price) come from the vast.ai listing screenshot and go
# in the conditions note, not into this file.
#
# STANDING RULE: the moment both files show 48, download them and DESTROY the
# instance in the same sitting. Never stop it: stopped instances keep charging.

set -u

OUT=/workspace/out
LOG=$OUT/sweep.log
mkdir -p "$OUT"

echo "== GPU check ======================================================"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU"
case "$GPU" in
  *"RTX 3090"*"24576 MiB"*) echo "OK: stock 24 GB RTX 3090" ;;
  *) echo "STOP: this is not a stock 24 GB RTX 3090. Destroy the instance and pick another."; exit 1 ;;
esac

echo "== setup =========================================================="
ln -sf "$(command -v python3)" /usr/local/bin/python
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git >/dev/null 2>&1
rm -rf /qvunex
git clone --depth 1 -q https://github.com/qaisermehdi3-coder/qvunex.git /qvunex

{
  echo "date_utc $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.memory --format=csv
  echo "--- cpu ---"; lscpu | grep -E "Model name|^CPU\(s\)"
  echo "nproc $(nproc)"
  echo "--- mem ---"; free -g | head -2
  echo "--- disk ---"; df -h /workspace | tail -1
} > "$OUT/host-3090-own.txt" 2>&1
cat "$OUT/host-3090-own.txt"

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

echo
echo "== pre-fetching all four models before any timing ================="
python3 - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-1.5B-Instruct",
          "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
          "Qwen/Qwen2.5-7B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct-AWQ"]:
    print("fetching", m, flush=True)
    snapshot_download(m)
    print("  done", flush=True)
PY
du -sh /workspace/hf/hub/models--* 2>/dev/null

echo
echo "== starting sweeps in background =================================="
nohup bash -c '
  set -u
  COMMON="--batches 1,8,32,128 --modes eager,graphs --max-model-len 1024 --repeats 3 --host-provider vast.ai_3090_own_see_note"
  echo "### 1.5B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ --compare-quantization awq \
    $COMMON --out /workspace/out/3090-own-1p5b.csv
  echo "### 1.5B done $(date -u)"
  echo "### 7B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ --compare-quantization awq \
    $COMMON --out /workspace/out/3090-own-7b.csv
  echo "### 7B done $(date -u)"
  cp /workspace/out/3090-own-*.csv /workspace/out/host-3090-own.txt /vllm-workspace/ 2>/dev/null
  echo "### copied for download $(date -u)"
' > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress:   tail -3 $LOG"
echo "count results:    grep -c \",ok,\" $OUT/3090-own-*.csv"
echo
echo "48 and 48 means done. Download the files, then DESTROY the instance."
