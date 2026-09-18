#!/usr/bin/env bash
# Second pass: eager worked, every failing config was cuda-graphs. Test graph
# capture on its own.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/debug.sh | bash

set -u

pkill -f sweep.py 2>/dev/null; sleep 1

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "== eager vs cuda-graphs, same model, same box ====================="
echo "weights are already cached, so this is quick."
echo

python3 - <<'PY'
import sys, traceback, time

def attempt(label, eager):
    from vllm import LLM, SamplingParams
    print(f"\n----- {label} (enforce_eager={eager}) -----", flush=True)
    t0 = time.time()
    try:
        llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
                  max_model_len=1024,
                  enforce_eager=eager,
                  gpu_memory_utilization=0.90,
                  enable_prefix_caching=False)
        out = llm.generate(["hello world"], SamplingParams(max_tokens=16))
        print(f"{label}: OK in {time.time()-t0:.1f}s", flush=True)
        return True
    except Exception:
        print(f"\n=== {label} FAILED after {time.time()-t0:.1f}s ===", flush=True)
        traceback.print_exc()
        return False

ok_eager = attempt("EAGER", True)
PY

echo
echo "now the same thing with graph capture, in a separate process"
echo "so a crash in one cannot mask the other:"

python3 - <<'PY'
import sys, traceback, time
from vllm import LLM, SamplingParams
print("\n----- CUDA GRAPHS (enforce_eager=False) -----", flush=True)
t0 = time.time()
try:
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              max_model_len=1024,
              enforce_eager=False,
              gpu_memory_utilization=0.90,
              enable_prefix_caching=False)
    out = llm.generate(["hello world"], SamplingParams(max_tokens=16))
    print(f"GRAPHS: OK in {time.time()-t0:.1f}s", flush=True)
except Exception:
    print(f"\n=== GRAPHS FAILED after {time.time()-t0:.1f}s ===", flush=True)
    traceback.print_exc()
    sys.exit(1)
PY
rc=$?

echo
echo "== verdict ========================================================"
if [ $rc -eq 0 ]; then
  echo "graphs work in the foreground. so the failure is in how sweep.py"
  echo "launches its child, not in the card. next suspect: the child's exit"
  echo "code, not its measurements."
else
  echo "graph capture itself fails on this host while eager works."
  echo "that is a finding, not a bug in the harness."
fi
