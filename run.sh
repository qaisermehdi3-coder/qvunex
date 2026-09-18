#!/usr/bin/env bash
# One-line bootstrap for the RTX 3090 leg of the GDDR-tier sweep.
#
# Run it on the rented vast.ai box, in the Jupyter terminal:
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/run.sh | bash
#
# It sets the box up, prints the facts the conditions note needs, then starts
# both model sizes in the background and returns immediately. Closing the
# browser does not kill it.
#
# Host values below are the provider's listing for vast.ai machine 84216, typed
# in by hand because a container's view of the host is not trustworthy.

set -u

OUT=/workspace/out
LOG=$OUT/sweep.log
mkdir -p "$OUT"

HOST_CPU="AMD EPYC 7K62 48-Core"
HOST_VCPUS="13.7"
HOST_RAM="36.8"
HOST_DISK="local NVMe"
HOST_PROVIDER="vast.ai machine 84216 host 443829 US"

echo "== setup =========================================================="
ln -sf "$(command -v python3)" /usr/local/bin/python
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git >/dev/null 2>&1
rm -rf /qvunex
git clone --depth 1 https://github.com/qaisermehdi3-coder/qvunex.git /qvunex

echo
echo "== facts for the conditions note =================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "--- disk ---"; df -h / | tail -1
echo "--- cpu  ---"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
echo "--- ram  ---"; free -g | head -2
echo "--- storage type (provider says: $HOST_DISK) ---"
lsblk -d -o NAME,ROTA,SIZE 2>/dev/null | head -5

echo
echo "== starting sweeps in background =================================="

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0

COMMON=(--batches 1,8,32,128 --modes eager,graphs
        --max-model-len 1024 --repeats 3
        --host-cpu "$HOST_CPU" --host-vcpus "$HOST_VCPUS"
        --host-ram-gb "$HOST_RAM" --host-disk "$HOST_DISK"
        --host-provider "$HOST_PROVIDER")

nohup bash -c "
  echo '### 1.5B starting' \$(date -u)
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
    --compare-quantization awq \
    ${COMMON[*]@Q} \
    --out $OUT/3090-1p5b.csv
  echo '### 1.5B done' \$(date -u)

  echo '### 7B starting' \$(date -u)
  python3 /qvunex/benchmarks/sweep.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --compare-quantization awq \
    ${COMMON[*]@Q} \
    --out $OUT/3090-7b.csv
  echo '### 7B done' \$(date -u)
" > "$LOG" 2>&1 &

echo "started. pid $!"
echo
echo "check progress any time with:"
echo "  tail -20 $LOG"
echo "count finished rows with:"
echo "  wc -l $OUT/*.csv"
