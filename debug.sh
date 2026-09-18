#!/usr/bin/env bash
# Find out why every config is failing.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/debug.sh | bash
#
# Kills the failing sweep first, then loads one model in the foreground with
# nothing swallowed, so the real traceback reaches the screen. sweep.py records
# only the tail of stderr, which on a killed process is the NCCL exit warning -
# true but useless.

set -u

echo "== stopping the failing sweep ====================================="
pkill -f sweep.py 2>/dev/null
pkill -f "1.5B starting" 2>/dev/null
sleep 2
echo "stopped."

echo
echo "== limits the container actually has =============================="
echo "--- cgroup memory limit (bytes; the number that matters) ---"
cat /sys/fs/cgroup/memory.max 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
  || echo "not readable"
echo "--- cgroup cpu limit ---"
cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "not readable"
echo "--- what free thinks (unreliable in a container) ---"
free -g | head -2
echo "--- gpu right now ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo
echo "== loading one model in the foreground ============================"
echo "smallest possible: 1.5B, eager, batch 1, 64 tokens."
echo "if this fails the traceback is the answer."
echo

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0

python3 - <<'PY'
import sys, traceback
print("python:", sys.version.split()[0], flush=True)
try:
    import torch, vllm
    print("torch :", torch.__version__, flush=True)
    print("vllm  :", vllm.__version__, flush=True)
    print("cuda  :", torch.version.cuda, "| available:", torch.cuda.is_available(), flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print("\nconstructing LLM ...", flush=True)
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_model_len=1024,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
    )
    print("LLM built. generating ...", flush=True)
    out = llm.generate(["hello world"], SamplingParams(max_tokens=16))
    print("OK:", out[0].outputs[0].text[:80], flush=True)
except Exception:
    print("\n=== THE ACTUAL ERROR ===", flush=True)
    traceback.print_exc()
    sys.exit(1)
PY

echo
echo "== done ==========================================================="
