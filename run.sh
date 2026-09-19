#!/usr/bin/env bash
# Card 2 of 4: RTX 4090 (stock 24GB), vast.ai machine 20082, datacenter 70142,
# Maryland US. Same conditions as the 3090 leg.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run.sh | bash
#
# HF_HUB_DISABLE_XET=1 is not optional. Without it the large AWQ shards never
# finish downloading - HuggingFace routes them through its Xet backend and that
# path fails from rented hosts with a connection error on /xet-read-token/.
# That cost a full day on the 3090 leg.

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
echo "VRAM must read 24576 MiB. If it says 49140 this is a modded 48GB card"
echo "and is NOT the RTX 4090 the client asked for - destroy it and rent again."

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

echo
echo "== pre-fetching all four models before any timing ================="
echo "downloading first means a slow or failed fetch cannot be mistaken"
echo "for a slow card. roughly 25GB."
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
    --host-cpu "AMD EPYC 7B13" \
    --host-vcpus "32.0" --host-ram-gb "129" \
    --host-disk "local NVMe" \
    --host-provider "vast.ai machine 20082 datacenter 70142 Maryland US" \
    --out /workspace/out/4090-1p5b.csv
  echo "### 1.5B done $(date -u)"

  echo "### 7B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON \
    --host-cpu "AMD EPYC 7B13" \
    --host-vcpus "32.0" --host-ram-gb "129" \
    --host-disk "local NVMe" \
    --host-provider "vast.ai machine 20082 datacenter 70142 Maryland US" \
    --out /workspace/out/4090-7b.csv
  echo "### 7B done $(date -u)"
' > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress:   tail -5 $LOG"
echo "count results:    grep -c \",ok,\" $OUT/4090-*.csv"
