#!/usr/bin/env bash
# Qvunex mixed-length test: what one long prompt does to the short answers
# already being streamed. Prepared 9 Nov 2026.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/mixed.sh | bash -s -- RATE MODEL SHORT_TOKENS LONG_TOKENS OUTPUT_TOKENS AT_ONCE LONG_EVERY SECONDS "CONFIGS"
#
# Only RATE (dollars per hour for the rented GPU) is required. Defaults:
#   MODEL Qwen/Qwen2.5-7B-Instruct, SHORT_TOKENS 512, LONG_TOKENS 12000,
#   OUTPUT_TOKENS 256, AT_ONCE 16 short requests in flight, LONG_EVERY 10 s,
#   SECONDS 60 per phase,
#   CONFIGS "default;--max-num-batched-tokens 16384" (vLLM flag sets, split
#   by ";", "default" = no extra flags; the server restarts for each).
#
# For each config, two phases of SECONDS each, all answers streamed:
#   short only: AT_ONCE short requests at a time (prompt ~SHORT_TOKENS)
#   mixed:      the same, plus one long request (prompt ~LONG_TOKENS) sent
#               every LONG_EVERY seconds on top
# Measured from the client per request: whole answer time, first token, and
# the biggest pause between two streamed tokens inside one answer (a long
# prompt that stalls everyone else shows up here). Cost per 1M output tokens
# = GPU price x window. Meter vs vLLM's own counters for every phase.
# The window runs until the last request started in it has finished.
#
# When it prints ALL DONE, download report-mixed.txt and mixed-files.tgz,
# then DESTROY the instance.

set -u
if [ -n "${QVUNEX_OUT:-}" ]; then OUT=$QVUNEX_OUT
elif [ -d /workspace ] && [ -w /workspace ]; then OUT=/workspace/mixed
else OUT="$HOME/qvunex-mixed"; fi
mkdir -p "$OUT" 2>/dev/null || OUT="$HOME/qvunex-mixed"
mkdir -p "$OUT" || { echo "cannot create an output folder; set QVUNEX_OUT"; exit 1; }
export QVUNEX_OUT="$OUT"

if [ "${MIXED_INNER:-}" != "1" ]; then
  curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/mixed.sh -o "$OUT/mixed.sh"
  MIXED_INNER=1 nohup bash "$OUT/mixed.sh" "$@" > "$OUT/run.out" 2>&1 &
  echo "started in background, pid $!  (report: $OUT/report.txt)"
  sleep 3
  exec tail -n +1 -F "$OUT/report.txt" 2>/dev/null
fi

RATE=${1:?usage: mixed.sh RATE [MODEL] [SHORT_TOKENS] [LONG_TOKENS] [OUTPUT_TOKENS] [AT_ONCE] [LONG_EVERY] [SECONDS] [CONFIGS]}
MODEL=${2:-Qwen/Qwen2.5-7B-Instruct}
SPTOK=${3:-512}
LPTOK=${4:-12000}
OTOK=${5:-256}
CONC=${6:-16}
EVERY=${7:-10}
SECS=${8:-60}
CONFIGS=${9:-default;--max-num-batched-tokens 16384}
PORT=${QVUNEX_PORT:-8000}
REP=$OUT/report.txt
: > "$REP"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ] && [ -w /workspace ]; then export HF_HOME=/workspace/hf; fi
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

say() { echo "$*" | tee -a "$REP"; }
now() { date +%s.%N; }

say "QVUNEX - MIXED PROMPT LENGTHS ON A RENTED GPU"
say "date_utc   $(date -u +%FT%TZ)"
GPUS=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)
NG=$(printf '%s\n' "$GPUS" | grep -c .)
G1=$(printf '%s\n' "$GPUS" | head -n 1)
if [ "$NG" -le 1 ]; then say "gpu        $G1"
elif [ "$(printf '%s\n' "$GPUS" | sort -u | grep -c .)" -eq 1 ]; then say "gpu        ${NG}x $G1"
else say "gpu        ${NG} GPUs $(printf '%s\n' "$GPUS" | cut -d, -f1 | paste -sd '+' -),$(echo "$G1" | cut -d, -f2-)"; fi
say "vllm       $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
say "cpu        $(lscpu | sed -n 's/^Model name: *//p' | head -1), nproc $(nproc)"
say "model      $MODEL"
say "load       $CONC short requests at once (prompt ~$SPTOK tokens); in the mixed phase also"
say "           one long request (prompt ~$LPTOK tokens) every ${EVERY}s; answers $OTOK tokens"
say "           (ignore_eos), streamed; ${SECS}s per phase"
say "configs    $CONFIGS"
say "price      \$$RATE per hour for this GPU"
say ""

curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/qvunex_single.py -o "$OUT/qvunex_single.py"
say "meter      qvunex_single.py md5 $(md5sum "$OUT/qvunex_single.py" | cut -c1-8)"
say "script     mixed.sh md5 $(md5sum "$OUT/mixed.sh" 2>/dev/null | cut -c1-8)"

t=$(now)
python3 - "$MODEL" <<'PY' >> "$OUT/run.out" 2>&1
import os, sys
if not os.path.isdir(sys.argv[1]):
    from huggingface_hub import snapshot_download
    snapshot_download(sys.argv[1])
PY
say "download   $(python3 -c "print(round($(now)-$t))") s (not counted in start-up)"
say ""

cat > "$OUT/mixload.py" <<'PY'
import json, sys, time, threading, uuid
sys.path.insert(0, sys.argv[6])
import qvunex_single as q
from openai import OpenAI
port, model, conc, secs, otok = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
sptok, lptok, every, mode, rawpath = int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9]), sys.argv[10], sys.argv[11]
client = q.wrap(OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x", timeout=600))
bodies = {"short": " ".join(["measure"] * sptok), "long": " ".join(["measure"] * lptok)}
recs, errs = [], {"short": 0, "long": 0}
first_err = [None]
lock = threading.Lock()
t0 = time.time()
end = t0 + secs

def one(kind):
    t = time.time()
    first = last = None
    gap = 0.0
    out = None
    try:
        stream = client.completions.create(model=model, prompt=f"{uuid.uuid4()} {bodies[kind]}",
                                           max_tokens=otok, temperature=0, stream=True,
                                           stream_options={"include_usage": True},
                                           extra_body={"ignore_eos": True})
        for ch in stream:
            n = time.time()
            if ch.choices:
                if first is None:
                    first = n
                else:
                    gap = max(gap, n - last)
                last = n
            if getattr(ch, "usage", None) is not None:
                out = ch.usage.completion_tokens
        done = time.time()
        with lock:
            recs.append({"kind": kind, "start": round(t - t0, 3), "latency": round(done - t, 4),
                         "first_token": None if first is None else round(first - t, 4),
                         "max_pause": round(gap, 4), "out_tokens": out})
    except Exception as e:
        with lock:
            errs[kind] += 1
            first_err[0] = first_err[0] or repr(e)[:200]

def short_worker():
    while time.time() < end:
        one("short")

threads = [threading.Thread(target=short_worker) for _ in range(conc)]
[x.start() for x in threads]
if mode == "mixed":
    k = 0
    while True:
        at = t0 + every / 2 + k * every
        if at >= end:
            break
        time.sleep(max(0.0, at - time.time()))
        lt = threading.Thread(target=one, args=("long",)); lt.start(); threads.append(lt)
        k += 1
[x.join() for x in threads]
el = time.time() - t0
with open(rawpath, "w") as f:
    for r in sorted(recs, key=lambda r: r["start"]):
        f.write(json.dumps(r) + "\n")

def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

def stats(kind):
    rs = [r for r in recs if r["kind"] == kind]
    known = [r["out_tokens"] for r in rs if r["out_tokens"] is not None]
    return {"done": len(rs), "errors": errs[kind],
            "out_tokens": sum(known) if len(known) == len(rs) else None,
            "lat_p50": pct([r["latency"] for r in rs], 0.5), "lat_p95": pct([r["latency"] for r in rs], 0.95),
            "ft_p50": pct([r["first_token"] for r in rs], 0.5), "ft_p95": pct([r["first_token"] for r in rs], 0.95),
            "ft_max": pct([r["first_token"] for r in rs], 1.0),
            "pause_p50": pct([r["max_pause"] for r in rs], 0.5), "pause_p95": pct([r["max_pause"] for r in rs], 0.95),
            "pause_max": pct([r["max_pause"] for r in rs], 1.0)}
print(json.dumps({"seconds": el, "short": stats("short"), "long": stats("long"), "first_err": first_err[0]}))
PY

cat > "$OUT/show.py" <<'PY'
import json, sys
name, phase, w, rate = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
try:
    r = json.loads(sys.argv[5])
except Exception:
    print(f"  {phase}: load script failed, see run.out"); sys.exit()
f = lambda x: "unknown" if x is None else f"{x:.2f} s"
f3 = lambda x: "unknown" if x is None else f"{x:.3f} s"
s, l = r["short"], r["long"]
outs = [x["out_tokens"] for x in (s, l) if x["done"]]
total = None if any(o is None for o in outs) else sum(outs)
cost = rate * w / 3600
print(f"  {phase}: window {w:.1f} s, " + (
    f"{total / w:,.0f} output tok/s, ${cost / total * 1e6:.3f} per 1M output tokens" if total
    else "output tokens unknown (no usage in the stream), cost per token unknown"))
print(f"    short answers   {s['done']} done, {s['errors']} errors; whole answer p50 {f(s['lat_p50'])} / p95 {f(s['lat_p95'])}")
print(f"                    first token p50 {f3(s['ft_p50'])} / p95 {f3(s['ft_p95'])}")
print(f"                    biggest pause inside an answer p50 {f3(s['pause_p50'])} / p95 {f3(s['pause_p95'])} / max {f3(s['pause_max'])}")
if phase.startswith("mixed"):
    print(f"    long prompts    {l['done']} done, {l['errors']} errors; first token p50 {f3(l['ft_p50'])} / max {f3(l['ft_max'])};"
          f" whole answer p50 {f(l['lat_p50'])}")
if r.get("first_err"):
    print(f"    first error     {r['first_err']}")
PY

cat > "$OUT/ttft.py" <<'PY'
import re, sys
def read(p):
    s = c = None
    try:
        lines = open(p)
    except OSError:
        return None, None
    for line in lines:
        m = re.match(r'vllm:time_to_first_token_seconds_(sum|count)\{[^}]*\} ([0-9.e+]+)\s*$', line)
        if m:
            if m.group(1) == "sum":
                s = (s or 0) + float(m.group(2))
            else:
                c = (c or 0) + float(m.group(2))
    return s, c
(bs, bc), (as_, ac) = read(sys.argv[1]), read(sys.argv[2])
if None in (bs, bc, as_, ac) or ac - bc <= 0:
    print("    server first token  not reported by the server (unknown, not zero)")
else:
    print(f"    server first token  mean {(as_ - bs) / (ac - bc):.3f} s over {ac - bc:.0f} requests, short and long together (vLLM's own histogram)")
PY

IFS=';' read -r -a CFGS <<< "$CONFIGS"
i=0
for cfg in "${CFGS[@]}"; do
  i=$((i+1))
  cfg=$(echo "$cfg" | sed 's/^ *//; s/ *$//')
  flags=$cfg; [ "$cfg" = "default" ] && flags=""
  say "CONFIG $i: ${cfg}"
  [ -d /vllm-workspace ] && pkill -f "bin/vllm serve" 2>/dev/null && sleep 5
  if curl -s --noproxy "*" -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
    say "  PORT $PORT IS ALREADY IN USE. Stop that server or set QVUNEX_PORT, then run again."; exit 1
  fi
  t0=$(now)
  nohup vllm serve "$MODEL" --port $PORT $flags > "$OUT/server_$i.log" 2>&1 &
  SPID=$!
  ok=1
  while true; do
    if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 $SPID 2>/dev/null; then
      say "  SERVER FAILED TO START with these flags. Last log lines:"
      grep -v -E "^\s*$" "$OUT/server_$i.log" | tail -n 8 | cut -c1-300 | sed 's/^/    /' | tee -a "$REP"
      ok=0; break
    fi
    sleep 1
  done
  [ $ok = 1 ] || { say ""; continue; }
  say "  start-up   $(python3 -c "print(round($(now)-$t0,1))") s"
  grep -h -i -E "Chunked prefill is enabled|max_num_batched_tokens=|GPU KV cache size" "$OUT/server_$i.log" | sed 's/^.*\] /  server     /' | sort -u | head -n 3 | tee -a "$REP"
  for phase in short mixed; do
    tag="${i}_$phase"
    curl -s --noproxy "*" "http://127.0.0.1:$PORT/metrics" -o "$OUT/before_$tag.txt"
    tb=$(now)
    res=$(QVUNEX_PATH="$OUT/events_$tag.jsonl" python3 "$OUT/mixload.py" $PORT "$MODEL" "$CONC" "$SECS" "$OTOK" "$OUT" "$SPTOK" "$LPTOK" "$EVERY" "$phase" "$OUT/requests_$tag.jsonl" 2>>"$OUT/run.out" | tail -n 1)
    curl -s --noproxy "*" "http://127.0.0.1:$PORT/metrics" -o "$OUT/after_$tag.txt"
    W=$(python3 -c "print(round($(now)-$tb,3))")
    echo "$res" > "$OUT/result_$tag.json"
    QVUNEX_PATH="$OUT/events_$tag.jsonl" python3 "$OUT/qvunex_single.py" --reconcile "$OUT/before_$tag.txt" "$OUT/after_$tag.txt" --gpu-rate "$RATE" --window "$W" > "$OUT/reconcile_$tag.txt" 2>&1
    label=$phase; [ $phase = short ] && label="short only"
    python3 "$OUT/show.py" "$cfg" "$label" "$W" "$RATE" "$res" | tee -a "$REP"
    grep -E "^\s+(requests|prompt tokens|output tokens)\s" "$OUT/reconcile_$tag.txt" | sed 's/^/    meter vs server /' >> "$REP"
    python3 "$OUT/ttft.py" "$OUT/before_$tag.txt" "$OUT/after_$tag.txt" >> "$REP"
  done
  kill $SPID 2>/dev/null; sleep 5
  [ -d /vllm-workspace ] && pkill -f "bin/vllm serve" 2>/dev/null && sleep 5
  say ""
done

say "WHAT THIS DOES NOT TELL YOU"
say "  One run per phase, synthetic prompts, long prompts at a fixed interval."
say "  Real traffic mixes many lengths and arrives in bursts. Costs assume the GPU"
say "  is busy the whole window."
say ""
cp "$REP" /vllm-workspace/report-mixed.txt 2>/dev/null
tar czf /vllm-workspace/mixed-files.tgz -C "$OUT" . 2>/dev/null
[ -d /vllm-workspace ] || tar czf "$OUT/mixed-files.tgz" -C "$OUT" --exclude=mixed-files.tgz . 2>/dev/null
say "Files: $REP and mixed-files.tgz (in /vllm-workspace if it exists, else in $OUT)"
say "ALL DONE. Download report-mixed.txt and mixed-files.tgz. On a rented GPU, DESTROY the instance."
