#!/usr/bin/env bash
# Two vLLM servers on ONE GPU: a small embedding model next to a chat model.
# Prepared 20 Oct 2026 to answer vllm-project/vllm#60599 with measured facts.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/coloc.sh | bash
#
# The tests run in the background; this command then follows the results.
# Closing the browser does not stop them. Everything lands in
# /workspace/coloc/results.txt and is copied to the Jupyter root at the end.
#
# What it checks:
#   A  embedding server alone, --gpu-memory-utilization 0.10
#   B  chat server started second, 0.70, both on the same GPU
#   C  throughput: each alone, then both busy at once (interference)
#   D  chat started first at 0.90, then the embedding server at 0.10
#   E  both started at the same moment (0.70 + 0.10)
#   F  how small the embedding server can go: 0.05, then 0.03
#
# STANDING RULE: when it prints ALL DONE, download results.txt and DESTROY
# the instance in the same sitting. Never stop it.

set -u
OUT=/workspace/coloc
mkdir -p "$OUT"

if [ "${COLOC_INNER:-}" != "1" ]; then
  curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/coloc.sh -o "$OUT/coloc.sh"
  COLOC_INNER=1 nohup bash "$OUT/coloc.sh" > "$OUT/run.out" 2>&1 &
  echo "started in background, pid $!  (results: $OUT/results.txt)"
  sleep 3
  exec tail -n +1 -F "$OUT/results.txt" 2>/dev/null
fi

RES=$OUT/results.txt
: > "$RES"
export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
EMB=BAAI/bge-small-en-v1.5
CHAT=Qwen/Qwen2.5-1.5B-Instruct
PE=8101
PC=8102

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RES"; }
gpu() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  gpu total: /' | tee -a "$RES"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  process: /' | tee -a "$RES"
}
start() { # name port model util [extra args]
  local name=$1 port=$2 model=$3 util=$4; shift 4
  say "start $name: vllm serve $model --gpu-memory-utilization $util $*"
  nohup vllm serve "$model" --port "$port" --gpu-memory-utilization "$util" "$@" > "$OUT/$name.log" 2>&1 &
  echo $! > "$OUT/$name.pid"
  date +%s > "$OUT/$name.t0"
}
ready() { # name port -> 0 when /health answers, 1 if the process died or 420 s passed
  local name=$1 port=$2 t0 pid
  t0=$(cat "$OUT/$name.t0"); pid=$(cat "$OUT/$name.pid")
  while true; do
    if curl -sf --noproxy "*" "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      say "$name READY after $(( $(date +%s) - t0 )) s"
      grep -h -i -E "KV cache memory|GPU KV cache size|Maximum concurrency" "$OUT/$name.log" | tail -n 3 | sed 's/^/  log: /' | tee -a "$RES"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      say "$name FAILED TO START after $(( $(date +%s) - t0 )) s. Last log lines:"
      grep -v -E "^\s*$" "$OUT/$name.log" | tail -n 12 | cut -c1-300 | sed 's/^/  log: /' | tee -a "$RES"
      return 1
    fi
    if [ $(( $(date +%s) - t0 )) -gt 420 ]; then
      say "$name NOT READY after 420 s"
      return 1
    fi
    sleep 2
  done
}
alive() { for f in "$OUT"/*.pid; do [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && return 0; done; pgrep -f "bin/vllm serve" >/dev/null; }
stop_all() {
  for f in "$OUT"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done
  pkill -f "bin/vllm serve" 2>/dev/null
  for i in $(seq 1 60); do alive || break; sleep 2; done
  for f in "$OUT"/*.pid; do [ -f "$f" ] && kill -9 "$(cat "$f")" 2>/dev/null; done
  pkill -9 -f "bin/vllm serve" 2>/dev/null
  sleep 8; rm -f "$OUT"/*.pid
  say "all servers stopped"; gpu
}

cat > "$OUT/load.py" <<'PY'
import json, sys, time, threading, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
mode, port, model, secs, conc = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with opener.open(r, timeout=300) as f:
        return json.load(f)
ok = err = units = 0
first_err = None
lock = threading.Lock()
end = time.time() + secs
def worker(i):
    global ok, err, units, first_err
    n = 0
    while time.time() < end:
        n += 1
        try:
            if mode == "embed":
                inp = [f"worker {i} request {n} sentence {k}: how much of this GPU sits idle" for k in range(32)]
                u = len(post("/v1/embeddings", {"model": model, "input": inp})["data"])
            else:
                r = post("/v1/completions", {"model": model, "prompt": f"{i}-{n} Explain what a GPU does.",
                                             "max_tokens": 128, "ignore_eos": True, "temperature": 0})
                u = r["usage"]["completion_tokens"]
            with lock:
                ok += 1; units += u
        except Exception as e:
            with lock:
                err += 1; first_err = first_err or repr(e)[:200]
t0 = time.time()
ts = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
[t.start() for t in ts]; [t.join() for t in ts]
el = time.time() - t0
what = "texts embedded" if mode == "embed" else "output tokens"
print(f"  {mode:5} {conc} at once, {el:.1f} s: {ok} requests ok, {err} errors, {units} {what} = {units/el:.1f} per s"
      + (f"  first error: {first_err}" if first_err else ""))
PY
load() { python3 "$OUT/load.py" "$@" | tee -a "$RES"; }

say "== host =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /' | tee -a "$RES"
python3 -c "import vllm; print('  vllm', vllm.__version__)" 2>/dev/null | tee -a "$RES"
echo "  nproc $(nproc)" | tee -a "$RES"
gpu
pkill -f "bin/vllm serve" 2>/dev/null

say "== fetching models =="
python3 - <<PY 2>&1 | tail -n 2
from huggingface_hub import snapshot_download
for m in ["$EMB", "$CHAT"]:
    snapshot_download(m)
print("models fetched")
PY

say "== A: embedding server alone at 0.10 =="
start emb $PE $EMB 0.10 --max-model-len 512
if ready emb $PE; then gpu; load embed $PE $EMB 30 8; fi

say "== B: chat server started second at 0.70, same GPU =="
start chat $PC $CHAT 0.70 --max-model-len 2048
if ready chat $PC; then
  gpu
  say "== C: throughput, each alone then both at once =="
  load embed $PE $EMB 30 8
  load chat $PC $CHAT 30 16
  say "  both busy at the same time:"
  python3 "$OUT/load.py" embed $PE $EMB 30 8 > "$OUT/c_emb.txt" &
  LP=$!
  python3 "$OUT/load.py" chat $PC $CHAT 30 16 > "$OUT/c_chat.txt"
  wait "$LP"
  cat "$OUT/c_emb.txt" "$OUT/c_chat.txt" | tee -a "$RES"
fi
stop_all

say "== D: chat first at 0.90, then embedding at 0.10 =="
start chat $PC $CHAT 0.90 --max-model-len 2048
if ready chat $PC; then
  gpu
  start emb $PE $EMB 0.10 --max-model-len 512
  ready emb $PE && gpu
fi
stop_all

say "== E: both started at the same moment (chat 0.70, embedding 0.10) =="
start chat $PC $CHAT 0.70 --max-model-len 2048
start emb $PE $EMB 0.10 --max-model-len 512
ready emb $PE
ready chat $PC
gpu
stop_all

say "== F: smallest embedding footprint =="
for u in 0.05 0.03; do
  start emb $PE $EMB $u --max-model-len 512 --max-num-seqs 32
  if ready emb $PE; then gpu; load embed $PE $EMB 20 8; fi
  stop_all
done

cp "$RES" /vllm-workspace/results-coloc.txt 2>/dev/null
tar czf /vllm-workspace/coloc-logs.tgz -C "$OUT" . 2>/dev/null
say "ALL DONE. Download results-coloc.txt and coloc-logs.tgz, then DESTROY the instance."
