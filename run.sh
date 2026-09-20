#!/usr/bin/env bash
# Card 4 of 4: NVIDIA L40S, vast.ai machine 150627, host 640216, Romania.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run.sh | bash
#
# Conditions-note caveats for this leg, all of which go in the writeup:
#   - 8 vCPUs. The other three legs had 13.7 (3090), 32 (4090) and 64 (A40).
#     Eight is the low end, and an outside measurement found the same RTX 4090
#     ran 1.85x slower in wall clock on a 5-vCPU host than a 24-vCPU one. So the
#     eager rows on this card may carry host-CPU cost that is not the card's.
#     CUDA-graph rows are far less sensitive to this; that is the whole point of
#     the reproducibility finding. Report the eager rows with that caveat.
#   - storage is a Samsung MZ7L, a SATA SSD at 2866 MB/s, not NVMe.
#   - host reliability 96.1%.
#   - only one L40S was listed on vast.ai at the time, so there was no
#     better-matched host to choose.

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
echo "GPU name must read NVIDIA L40S. Stop if it says anything else."

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
    --host-cpu "Intel Xeon Silver 4-series" \
    --host-vcpus "8.0" --host-ram-gb "64" \
    --host-disk "Samsung MZ7L SATA SSD 2866 MB/s" \
    --host-provider "vast.ai machine 150627 host 640216 Romania" \
    --out /workspace/out/l40s-1p5b.csv
  echo "### 1.5B done $(date -u)"

  echo "### 7B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON \
    --host-cpu "Intel Xeon Silver 4-series" \
    --host-vcpus "8.0" --host-ram-gb "64" \
    --host-disk "Samsung MZ7L SATA SSD 2866 MB/s" \
    --host-provider "vast.ai machine 150627 host 640216 Romania" \
    --out /workspace/out/l40s-7b.csv
  echo "### 7B done $(date -u)"
' > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress:   tail -5 $LOG"
echo "count results:    grep -c \",ok,\" $OUT/l40s-*.csv"
