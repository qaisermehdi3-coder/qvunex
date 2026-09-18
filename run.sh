#!/usr/bin/env bash
# Clean re-run for the RTX 3090, after two problems were found and fixed:
#
#   1. fp16 weights were not cached on the first attempt, so every config in
#      that run failed. They are cached now.
#   2. Qwen2.5-7B-Instruct-AWQ never finished downloading. HuggingFace routes
#      large files through its Xet backend and that path fails from this host:
#         ConnectionError ... in xet_get ... /xet-read-token/
#      HF_HUB_DISABLE_XET=1 forces the plain HTTP path and it completed, 4.00GB.
#
# Everything is on local disk now, so this run measures and nothing downloads.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run.sh | bash

set -u

OUT=/workspace/out
LOG=$OUT/sweep-v2.log
mkdir -p "$OUT"

HOST_CPU="AMD EPYC 7K62 48-Core"
HOST_VCPUS="13.7"
HOST_RAM="36.8"
HOST_DISK="local NVMe"
HOST_PROVIDER="vast.ai machine 84216 host 443829 US"

echo "== refreshing the repo ============================================"
rm -rf /qvunex
git clone --depth 1 -q https://github.com/qaisermehdi3-coder/qvunex.git /qvunex
ln -sf "$(command -v python3)" /usr/local/bin/python

echo "== cached weights (nothing should download) ======================="
du -sh /workspace/hf/hub/models--* 2>/dev/null

echo
echo "== starting clean sweeps in background ============================"

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1        # the fix
export HF_HUB_OFFLINE=0

nohup bash -c '
  set -u
  COMMON="--batches 1,8,32,128 --modes eager,graphs --max-model-len 1024 --repeats 3"

  echo "### 1.5B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON \
    --host-cpu "AMD EPYC 7K62 48-Core" \
    --host-vcpus "13.7" --host-ram-gb "36.8" \
    --host-disk "local NVMe" \
    --host-provider "vast.ai machine 84216 host 443829 US" \
    --out /workspace/out/3090-1p5b-v2.csv
  echo "### 1.5B done $(date -u)"

  echo "### 7B starting $(date -u)"
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --compare-quantization awq \
    $COMMON \
    --host-cpu "AMD EPYC 7K62 48-Core" \
    --host-vcpus "13.7" --host-ram-gb "36.8" \
    --host-disk "local NVMe" \
    --host-provider "vast.ai machine 84216 host 443829 US" \
    --out /workspace/out/3090-7b-v2.csv
  echo "### 7B done $(date -u)"
' > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress:   tail -5 $LOG"
echo "count results:    grep -c \",ok,\" $OUT/*v2.csv"
