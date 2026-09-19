#!/usr/bin/env bash
# Card 3 of 4: NVIDIA A40 (48GB), vast.ai machine 151322, host 682584, Belgium.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run.sh | bash
#
# Notes for the conditions note on this host:
#   - host reliability listed at 93.0%, lower than the 3090 (98.9%) and
#     4090 (99.89%) legs. If the box dies mid-sweep, sweep.py rewrites its CSV
#     after every config, so completed rows survive.
#   - storage is a Toshiba SSD at 2583 MB/s, not the NVMe the other two had.
#   - 64 vCPUs against 13.7 on the 3090 and 32 on the 4090. Recorded per row.
#
# HF_HUB_DISABLE_XET=1 is mandatory: without it the large AWQ shards never
# finish downloading.

set -u

OUT=/workspace/out
LOG=$OUT/sweep.log
mkdir -p "$OUT"

echo "== setup =========================================================="
ln -sf "$(command -v python3)" /usr/local/bin/python
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git >/dev/null 2>&1
rm -rf /qvunex
git clone --depth 1 -q https://github.com/qaisermehdi3-coder/qvunex.git /qvunex

echo
echo "== facts for the conditions note =================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "--- disk ---"; df -h / | tail -1
echo "--- cpu  ---"; lscpu | grep -E "Model name|^CPU\(s\)"
echo
echo "GPU name must read NVIDIA A40. Stop if it says anything else."

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

echo
echo "== cached sizes ==================================================="
du -sh /workspace/hf/hub/models--* 2>/dev/null

echo
echo "== starting sweeps in background =================================="

nohup bash -c '
  set -u
  COMMON="--batches 1,8,32,128 --modes eager,graphs --max-model-len 1024 --repeats 3"

  echo "### 1.5B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON \
    --host-cpu "AMD EPYC 7662" \
    --host-vcpus "64.0" --host-ram-gb "513" \
    --host-disk "Toshiba SSD 2583 MB/s" \
    --host-provider "vast.ai machine 151322 host 682584 Belgium" \
    --out /workspace/out/a40-1p5b.csv
  echo "### 1.5B done $(date -u)"

  echo "### 7B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON \
    --host-cpu "AMD EPYC 7662" \
    --host-vcpus "64.0" --host-ram-gb "513" \
    --host-disk "Toshiba SSD 2583 MB/s" \
    --host-provider "vast.ai machine 151322 host 682584 Belgium" \
    --out /workspace/out/a40-7b.csv
  echo "### 7B done $(date -u)"
' > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress:   tail -5 $LOG"
echo "count results:    grep -c \",ok,\" $OUT/a40-*.csv"
