#!/usr/bin/env bash
# Third pass. Card is fine, graphs are fine, so the fault is in how sweep.py
# runs its child. Two things: show me the launching code, then run exactly one
# config in the foreground with nothing swallowed.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/debug.sh | bash

set -u

pkill -f sweep.py 2>/dev/null; sleep 1

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "== how the child is launched ======================================"
grep -n "subprocess\|sys.executable\|\"python\"\|'python'\|returncode\|--single" \
     /qvunex/benchmarks/sweep.py | head -25

echo
echo "== what 'python' resolves to ======================================"
which python || echo "no python on PATH  <-- would explain everything"
which python3
readlink -f /usr/local/bin/python 2>/dev/null || echo "symlink missing"

echo
echo "== one config, foreground, nothing hidden ========================="
python3 /qvunex/benchmarks/sweep.py \
  --batches 1 --modes graphs --repeats 1 \
  --max-model-len 1024 --out /tmp/one.csv
echo "sweep.py exit code: $?"

echo
echo "== the row it wrote ==============================================="
tail -2 /tmp/one.csv 2>/dev/null || echo "no csv written"
