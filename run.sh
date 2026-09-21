#!/usr/bin/env bash
# L40S — Qaiser's OWN run, 21 September 2026. Not client work.
# vast.ai instance 51944671, machine 150627, host 640216, Romania.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run.sh | bash
#
# Why this run exists: the L40S already in the corpus belongs to a client and
# cannot be published without their say-so. This one is measured on the same
# script and the same flags, on Qaiser's own account, so the numbers are his to
# publish. It is the same physical machine (150627), so it is a card he owns
# outright, not an independent second host. That limit goes in the writeup.
#
# Conditions for the note:
#   - 8.0 of 64 vCPU, Intel Xeon Silver 4514Y. Eight is the low end of the range
#     where the host-CPU effect is strongest, so the eager rows may carry host
#     cost that is not the card's. CUDA-graph rows are far less sensitive.
#   - storage Samsung MZ7L37T6, SATA SSD at 2947 MB/s, not NVMe.
#   - host reliability 97.2%.
#   - only one L40S was listed on vast.ai at the time, again.
#
# STANDING RULE: the moment grep shows 48/48, download the CSVs and DESTROY the
# instance in the same sitting. Never stop it — stopped instances keep charging
# for storage.

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
  HOST="--host-cpu Intel_Xeon_Silver_4514Y --host-vcpus 8.0 --host-ram-gb 64 --host-disk Samsung_MZ7L37T6_SATA_2947MBps --host-provider vast.ai_m150627_h640216_Romania_own"

  echo "### 1.5B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON $HOST \
    --out /workspace/out/l40s-own-1p5b.csv
  echo "### 1.5B done $(date -u)"

  echo "### 7B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON $HOST \
    --out /workspace/out/l40s-own-7b.csv
  echo "### 7B done $(date -u)"
' > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress:   tail -5 $LOG"
echo "count results:    grep -c \",ok,\" $OUT/l40s-own-*.csv"
echo
echo "48 and 48 means done. Download both CSVs, then DESTROY the instance."
