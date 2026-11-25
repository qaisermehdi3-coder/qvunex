#!/usr/bin/env bash
# Qvunex long-pair test: do two long prompts sent together starve each other's
# decode? Prepared 26 Nov 2026 for vllm-project/vllm#54919.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/longpair.sh | bash -s -- RATE MODEL "FLAGS" LONG_A LONG_B MAX_TOKENS
#
# Only RATE (dollars per hour for the rented GPU) is required. Defaults follow
# the repro in vllm#54919 (yuhinso):
#   MODEL cyankiwi/Qwen3.8-27B-AWQ-INT4
#   FLAGS "--max-model-len 200000 --max-num-seqs 3 --max-num-batched-tokens 4096
#          --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --enable-prefix-caching"
#   LONG_A 60000, LONG_B 75000 (prompt tokens, about one per word), MAX_TOKENS 1024
#
# Phases, every answer streamed and timed token by token:
#   control: two ~500-token prompts sent together
#   long pair: two long prompts (LONG_A, LONG_B) sent together, each with its
#              own random prefix so no prefix-cache reuse
# For each request: first token, every later token's time, and the longest gap.
# The key number: how fast one request decodes while the other is still being
# read (its first token not yet out). /metrics is saved every 3 s.
#
# When it prints ALL DONE, download report-longpair.txt and longpair-files.tgz,
# then DESTROY the instance.

set -u
if [ -n "${QVUNEX_OUT:-}" ]; then OUT=$QVUNEX_OUT
elif [ -d /workspace ] && [ -w /workspace ]; then OUT=/workspace/longpair
else OUT="$HOME/qvunex-longpair"; fi
mkdir -p "$OUT" 2>/dev/null || OUT="$HOME/qvunex-longpair"
mkdir -p "$OUT" || { echo "cannot create an output folder; set QVUNEX_OUT"; exit 1; }
export QVUNEX_OUT="$OUT"

if [ "${LONGPAIR_INNER:-}" != "1" ]; then
  curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/longpair.sh -o "$OUT/longpair.sh"
  LONGPAIR_INNER=1 nohup bash "$OUT/longpair.sh" "$@" > "$OUT/run.out" 2>&1 &
  echo "started in background, pid $!  (report: $OUT/report.txt)"
  sleep 3
  exec tail -n +1 -F "$OUT/report.txt" 2>/dev/null
fi

RATE=${1:?usage: longpair.sh RATE [MODEL] [FLAGS] [LONG_A] [LONG_B] [MAX_TOKENS]}
MODEL=${2:-cyankiwi/Qwen3.8-27B-AWQ-INT4}
FLAGS=${3:---max-model-len 200000 --max-num-seqs 3 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --enable-prefix-caching}
LA=${4:-60000}
LB=${5:-75000}
MAXTOK=${6:-1024}
PORT=${QVUNEX_PORT:-8000}
REP=$OUT/report.txt
: > "$REP"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ] && [ -w /workspace ]; then export HF_HOME=/workspace/hf; fi
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

say() { echo "$*" | tee -a "$REP"; }
now() { date +%s.%N; }

say "QVUNEX - TWO LONG PROMPTS AT ONCE ON A RENTED GPU (vllm#54919)"
say "date_utc   $(date -u +%FT%TZ)"
say "gpu        $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
say "vllm       $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
say "cpu        $(lscpu | sed -n 's/^Model name: *//p' | head -1), nproc $(nproc)"
say "model      $MODEL"
say "flags      $FLAGS"
say "load       control: 2 x ~500-token prompts together; long pair: ~$LA and ~$LB tokens"
say "           together; answers $MAXTOK tokens (ignore_eos, temperature 0), streamed"
say "price      \$$RATE per hour for this GPU"
say ""

curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/qvunex_single.py -o "$OUT/qvunex_single.py"
say "meter      qvunex_single.py md5 $(md5sum "$OUT/qvunex_single.py" | cut -c1-8)"
say "script     longpair.sh md5 $(md5sum "$OUT/longpair.sh" 2>/dev/null | cut -c1-8)"

t=$(now)
python3 - "$MODEL" <<'PY' >> "$OUT/run.out" 2>&1
import os, sys
if not os.path.isdir(sys.argv[1]):
    from huggingface_hub import snapshot_download
    snapshot_download(sys.argv[1])
PY
say "download   $(python3 -c "print(round($(now)-$t))") s (not counted in start-up)"

cat > "$OUT/budget.py" <<'PY'
import sys
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
a = make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])
if getattr(a, "model_tag", None):
    a.model = a.model_tag
c = AsyncEngineArgs.from_cli_args(a).create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
s = c.scheduler_config
print(f"budget     max_num_batched_tokens {s.max_num_batched_tokens}, chunked prefill "
      f"{'on' if s.enable_chunked_prefill else 'off'}, max_num_seqs {s.max_num_seqs}, "
      f"async scheduling {getattr(s, 'async_scheduling', 'unknown')} "
      "(from vLLM's own EngineArgs for these flags, not from the server log)")
PY

cat > "$OUT/pair.py" <<'PY'
import json, sys, time, threading, uuid
sys.path.insert(0, sys.argv[4])
import qvunex_single as q
from openai import OpenAI
port, model, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3])
la, lb, rawpath = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
client = q.wrap(OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x", timeout=3600))
recs = {}
lock = threading.Lock()
t0 = time.time()

def one(name, n):
    t = time.time()
    stamps, out, err = [], None, None
    try:
        stream = client.completions.create(model=model, prompt=f"{uuid.uuid4()} " + " ".join(["measure"] * n),
                                           max_tokens=maxtok, temperature=0, stream=True,
                                           stream_options={"include_usage": True},
                                           extra_body={"ignore_eos": True})
        for ch in stream:
            if ch.choices:
                stamps.append(time.time() - t0)
            if getattr(ch, "usage", None) is not None:
                out = ch.usage.completion_tokens
    except Exception as e:
        err = repr(e)[:200]
    with lock:
        recs[name] = {"start": t - t0, "end": time.time() - t0, "stamps": stamps, "out_tokens": out,
                      "prompt_words": n, "error": err}

ts = [threading.Thread(target=one, args=("A", la)), threading.Thread(target=one, args=("B", lb))]
[x.start() for x in ts]; [x.join() for x in ts]
json.dump(recs, open(rawpath, "w"))

def rate_between(r, a, b):
    n = sum(1 for s in r["stamps"] if a <= s < b)
    return (n / (b - a)) if b > a else None

res = {}
for name, r in recs.items():
    other = recs["B" if name == "A" else "A"]
    st = r["stamps"]
    first = st[0] if st else None
    gaps = [st[i + 1] - st[i] for i in range(len(st) - 1)]
    o_first = other["stamps"][0] if other["stamps"] else None
    # while the other request is still being read (its first token not yet out)
    # only windows of at least 1 s, so a few tokens cannot look like a rate
    w = rate_between(r, first, o_first) if (first is not None and o_first is not None and o_first - first >= 1.0) else None
    after_a = max(first, o_first) if (first is not None and o_first is not None) else first
    after = rate_between(r, after_a, st[-1]) if (st and after_a is not None and st[-1] > after_a) else None
    res[name] = {"prompt_words": r["prompt_words"], "first_token": None if first is None else round(first - r["start"], 3),
                 "done": round(r["end"] - r["start"], 2), "tokens": len(st), "out_tokens": r["out_tokens"],
                 "max_gap": round(max(gaps), 3) if gaps else None,
                 "rate_while_other_prefills": None if w is None else round(w, 2),
                 "window_s": None if w is None else round(o_first - first, 2),
                 "rate_after": None if after is None else round(after, 2), "error": r["error"]}
wall = max(r["end"] for r in recs.values()) - min(r["start"] for r in recs.values())
outs = [r["out_tokens"] for r in recs.values()]
res["wall"] = round(wall, 2)
res["aggregate_tok_s"] = None if any(o is None for o in outs) else round(sum(outs) / wall, 2)
print(json.dumps(res))
PY

cat > "$OUT/poll.sh" <<'SH'
#!/usr/bin/env bash
# save /metrics every 3 s until the stop file appears
i=0
while [ ! -f "$2/stop_poll" ]; do
  curl -s --noproxy "*" "http://127.0.0.1:$1/metrics" -o "$2/poll/m_$(printf %04d $i)_$(date +%s).txt"
  i=$((i+1)); sleep 3
done
SH

cat > "$OUT/show.py" <<'PY'
import json, sys
phase = sys.argv[1]
try:
    r = json.loads(sys.argv[2])
except Exception:
    print(f"  {phase}: script failed, see run.out"); sys.exit()
f = lambda x, u="": "unknown" if x is None else f"{x}{u}"
print(f"  {phase}: wall {r['wall']} s, aggregate {f(r['aggregate_tok_s'], ' output tok/s')}")
for k in ("A", "B"):
    x = r[k]
    print(f"    request {k} (prompt ~{x['prompt_words']} tokens): first token {f(x['first_token'], ' s')}, done {x['done']} s, "
          f"{x['tokens']} tokens streamed, longest gap {f(x['max_gap'], ' s')}")
    if x["rate_while_other_prefills"] is None and x["first_token"] is not None:
        w = "n/a (the other request's first token came first, or less than 1 s apart)"
    else:
        w = f"{f(x['rate_while_other_prefills'], ' tok/s')} (over {f(x['window_s'], ' s')})"
    print(f"      decode while the other is still being read: {w}; after: {f(x['rate_after'], ' tok/s')}")
    if x["error"]:
        print(f"      error: {x['error']}")
PY

cfg=$(echo "$FLAGS" | sed 's/^ *//; s/ *$//')
B=$(timeout 300 python3 "$OUT/budget.py" "$MODEL" $cfg 2>>"$OUT/run.out" | tail -n 1)
case "$B" in budget*) say "$B" ;; *) say "budget     unknown (could not compute; see run.out)" ;; esac
[ -d /vllm-workspace ] && pkill -f "bin/vllm serve" 2>/dev/null && sleep 5
if curl -s --noproxy "*" -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
  say "PORT $PORT IS ALREADY IN USE. Stop that server or set QVUNEX_PORT, then run again."; exit 1
fi
t0=$(now)
nohup vllm serve "$MODEL" --port $PORT $cfg > "$OUT/server.log" 2>&1 &
SPID=$!
while true; do
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then
    say "SERVER FAILED TO START with these flags. Last log lines:"
    grep -v -E "^\s*$" "$OUT/server.log" | tail -n 12 | cut -c1-300 | sed 's/^/    /' | tee -a "$REP"
    tar czf /vllm-workspace/longpair-files.tgz -C "$OUT" . 2>/dev/null
    say "ALL DONE (server did not start). Download report-longpair.txt and longpair-files.tgz."
    cp "$REP" /vllm-workspace/report-longpair.txt 2>/dev/null
    exit 1
  fi
  sleep 1
done
say "start-up   $(python3 -c "print(round($(now)-$t0,1))") s"
grep -h -i -E "GPU KV cache size|Maximum concurrency" "$OUT/server.log" | sed 's/^.*\] /server     /' | sort -u | head -n 3 | tee -a "$REP"
say ""

mkdir -p "$OUT/poll"
bash "$OUT/poll.sh" $PORT "$OUT" &
PPID2=$!
for phase in control longpair; do
  if [ $phase = control ]; then a=500; b=500; label="control (2 x ~500 tokens)"; else a=$LA; b=$LB; label="long pair"; fi
  curl -s --noproxy "*" "http://127.0.0.1:$PORT/metrics" -o "$OUT/before_$phase.txt"
  tb=$(now)
  res=$(QVUNEX_PATH="$OUT/events_$phase.jsonl" python3 "$OUT/pair.py" $PORT "$MODEL" "$MAXTOK" "$OUT" "$a" "$b" "$OUT/requests_$phase.json" 2>>"$OUT/run.out" | tail -n 1)
  curl -s --noproxy "*" "http://127.0.0.1:$PORT/metrics" -o "$OUT/after_$phase.txt"
  W=$(python3 -c "print(round($(now)-$tb,3))")
  echo "$res" > "$OUT/result_$phase.json"
  QVUNEX_PATH="$OUT/events_$phase.jsonl" python3 "$OUT/qvunex_single.py" --reconcile "$OUT/before_$phase.txt" "$OUT/after_$phase.txt" --gpu-rate "$RATE" --window "$W" > "$OUT/reconcile_$phase.txt" 2>&1
  python3 "$OUT/show.py" "$label" "$res" | tee -a "$REP"
  grep -E "^\s+(requests|prompt tokens|output tokens)\s" "$OUT/reconcile_$phase.txt" | sed 's/^/    meter vs server /' >> "$REP"
  say ""
done
touch "$OUT/stop_poll"; sleep 4; kill $PPID2 2>/dev/null
kill $SPID 2>/dev/null; sleep 5
[ -d /vllm-workspace ] && pkill -f "bin/vllm serve" 2>/dev/null

say "WHAT THIS DOES NOT TELL YOU"
say "  One run, one pair of prompts, synthetic text (a repeated word), completions API"
say "  with ignore_eos (the #54919 repro used chat completions without it)."
say ""
cp "$REP" /vllm-workspace/report-longpair.txt 2>/dev/null
tar czf /vllm-workspace/longpair-files.tgz -C "$OUT" . 2>/dev/null
[ -d /vllm-workspace ] || tar czf "$OUT/longpair-files.tgz" -C "$OUT" --exclude=longpair-files.tgz . 2>/dev/null
say "Files: $REP and longpair-files.tgz (in /vllm-workspace if it exists, else in $OUT)"
say "ALL DONE. Download report-longpair.txt and longpair-files.tgz. On a rented GPU, DESTROY the instance."
