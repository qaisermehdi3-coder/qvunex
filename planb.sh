#!/usr/bin/env bash
# Qvunex "your config on our GPU" report. Prepared 25 Oct 2026.
#
# For an operator who will not run a stranger's file on their own servers:
# they send their model, vLLM flags, typical prompt/answer length and GPU
# price, we rent the same GPU, run this, and send them the report.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/planb.sh | bash -s -- RATE MODEL PROMPT_TOKENS OUTPUT_TOKENS "LEVELS" SECONDS "EXTRA_VLLM_FLAGS"
#
# Only RATE (dollars per hour for the rented GPU) is required. Defaults:
#   MODEL Qwen/Qwen2.5-7B-Instruct, PROMPT_TOKENS 512, OUTPUT_TOKENS 256,
#   LEVELS "1 4 16 64" (requests in flight at once), SECONDS 60 per level,
#   EXTRA_VLLM_FLAGS "" (for example "--max-model-len 4096").
#
# What it measures, on this one rented card:
#   - start-up: seconds from launch to /health answering (model already on disk)
#   - for each level: requests finished, output tokens/s, latency p50/p95, and
#     the GPU cost per request and per 1M output tokens AT THAT LOAD, through
#     qvunex_single.py --reconcile (meter vs vLLM's own counters, --gpu-rate)
# What it cannot measure: how much of a real day the server sits idle. That
# needs their own traffic (TRY-IT.md, or the paid run on their servers).
#
# The tests run in the background; this command then follows the report.
# When it prints ALL DONE, download report-planb.txt and DESTROY the instance.
# On your own machine it writes to ~/qvunex-planb (or QVUNEX_OUT if set).
# It also writes corpus-rows.csv: the same numbers as rows for
# benchmarks/corpus/served-runs.csv (host and source are left for you to fill).

set -u
# Where the files go: QVUNEX_OUT if set, else /workspace/planb (vast.ai and
# similar), else ~/qvunex-planb on a normal machine.
OUT=${QVUNEX_OUT:-/workspace/planb}
mkdir -p "$OUT" 2>/dev/null || OUT="$HOME/qvunex-planb"
mkdir -p "$OUT" || { echo "cannot create an output folder; set QVUNEX_OUT"; exit 1; }
export QVUNEX_OUT="$OUT"

if [ "${PLANB_INNER:-}" != "1" ]; then
  curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/planb.sh -o "$OUT/planb.sh"
  PLANB_INNER=1 nohup bash "$OUT/planb.sh" "$@" > "$OUT/run.out" 2>&1 &
  echo "started in background, pid $!  (report: $OUT/report.txt)"
  sleep 3
  exec tail -n +1 -F "$OUT/report.txt" 2>/dev/null
fi

RATE=${1:?usage: planb.sh RATE [MODEL] [PROMPT_TOKENS] [OUTPUT_TOKENS] [LEVELS] [SECONDS] [EXTRA]}
MODEL=${2:-Qwen/Qwen2.5-7B-Instruct}
PTOK=${3:-512}
OTOK=${4:-256}
LEVELS=${5:-1 4 16 64}
SECS=${6:-60}
EXTRA=${7:-}
PORT=${QVUNEX_PORT:-8000}
REP=$OUT/report.txt
: > "$REP"
export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

say() { echo "$*" | tee -a "$REP"; }
now() { date +%s.%N; }

say "QVUNEX - YOUR CONFIG ON A RENTED GPU"
say "date_utc   $(date -u +%FT%TZ)"
say "gpu        $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
say "vllm       $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
say "cpu        $(lscpu | sed -n 's/^Model name: *//p' | head -1), nproc $(nproc)"
say "model      $MODEL"
say "flags      ${EXTRA:-none}"
say "load       prompt ~$PTOK tokens, answer $OTOK tokens (ignore_eos), levels: $LEVELS, ${SECS}s each"
say "price      \$$RATE per hour for this GPU"
say ""

curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/qvunex_single.py -o "$OUT/qvunex_single.py"
say "meter      qvunex_single.py md5 $(md5sum "$OUT/qvunex_single.py" | cut -c1-8)"

t=$(now)
python3 - "$MODEL" <<'PY' >> "$OUT/run.out" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
PY
say "download   $(python3 -c "print(round($(now)-$t))") s (not counted in start-up)"

# On a rented box (/workspace) clear leftovers from an earlier run. On your own
# machine nothing else is stopped: the port must be free instead.
[ "$OUT" = /workspace/planb ] && pkill -f "bin/vllm serve" 2>/dev/null && sleep 5
if curl -s --noproxy "*" -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
  say "PORT $PORT IS ALREADY IN USE. Stop that server or set QVUNEX_PORT to a free port, then run again."
  exit 1
fi
t0=$(now)
nohup vllm serve "$MODEL" --port $PORT $EXTRA > "$OUT/server.log" 2>&1 &
SPID=$!
while true; do
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then
    say "SERVER FAILED TO START. Last log lines:"
    grep -v -E "^\s*$" "$OUT/server.log" | tail -n 15 | cut -c1-300 | tee -a "$REP"
    exit 1
  fi
  sleep 1
done
BOOT=$(python3 -c "print(round($(now)-$t0,1))")
say "start-up   $BOOT s from launch to /health (model already on disk)"
grep -h -i -E "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$OUT/server.log" | tail -n 3 | sed 's/^.*\] /           /' | tee -a "$REP"
say ""

cat > "$OUT/load.py" <<'PY'
import json, sys, time, threading, uuid
sys.path.insert(0, sys.argv[6])
import qvunex_single as q
from openai import OpenAI
port, model, conc, secs, otok = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
ptok = int(sys.argv[7])
client = q.wrap(OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x"))
body = " ".join(["measure"] * ptok)
lat, ok, err, out_tokens = [], 0, 0, 0
first_err = None
lock = threading.Lock()
end = time.time() + secs
def worker():
    global ok, err, out_tokens, first_err
    while time.time() < end:
        t = time.time()
        try:
            r = client.completions.create(model=model, prompt=f"{uuid.uuid4()} {body}",
                                          max_tokens=otok, temperature=0,
                                          extra_body={"ignore_eos": True})
            with lock:
                ok += 1; lat.append(time.time() - t); out_tokens += r.usage.completion_tokens
        except Exception as e:
            with lock:
                err += 1; first_err = first_err or repr(e)[:200]
t0 = time.time()
ts = [threading.Thread(target=worker) for _ in range(conc)]
[x.start() for x in ts]; [x.join() for x in ts]
el = time.time() - t0
lat.sort()
pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float("nan")
print(json.dumps({"ok": ok, "err": err, "seconds": el, "out_tokens": out_tokens,
                  "p50": pct(0.5), "p95": pct(0.95), "first_err": first_err}))
PY

say "PER LOAD LEVEL (server busy the whole window; cost = GPU price x window)"
say ""
for c in $LEVELS; do
  curl -s --noproxy "*" "http://127.0.0.1:$PORT/metrics" -o "$OUT/before_$c.txt"
  tb=$(now)
  res=$(QVUNEX_PATH="$OUT/events_$c.jsonl" python3 "$OUT/load.py" $PORT "$MODEL" "$c" "$SECS" "$OTOK" "$OUT" "$PTOK" 2>>"$OUT/run.out" | tail -n 1)
  curl -s --noproxy "*" "http://127.0.0.1:$PORT/metrics" -o "$OUT/after_$c.txt"
  W=$(python3 -c "print(round($(now)-$tb,3))")
  QVUNEX_PATH="$OUT/events_$c.jsonl" python3 "$OUT/qvunex_single.py" --reconcile "$OUT/before_$c.txt" "$OUT/after_$c.txt" --gpu-rate "$RATE" --window "$W" > "$OUT/reconcile_$c.txt" 2>&1
  python3 - "$c" "$W" "$RATE" "$res" <<'PY' | tee -a "$REP"
import json, sys
c, w, rate = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    r = json.loads(sys.argv[4])
except Exception:
    print(f"  {c:>3} at once: load script failed, see run.out"); sys.exit()
cost = rate * w / 3600
per_req = cost / r["ok"] if r["ok"] else None
per_m = cost / r["out_tokens"] * 1e6 if r["out_tokens"] else None
s = (f"  {c:>3} at once: {r['ok']} done, {r['err']} errors, "
     f"{r['out_tokens'] / w:,.0f} output tok/s, latency p50 {r['p50']:.2f} s / p95 {r['p95']:.2f} s, "
     f"${per_req:.6f} per request, ${per_m:.3f} per 1M output tokens" if per_req else
     f"  {c:>3} at once: 0 requests done, {r['err']} errors ({r['first_err']})")
print(s)
PY
  grep -E "^\s+(requests|prompt tokens|output tokens)\s" "$OUT/reconcile_$c.txt" | sed 's/^/        meter vs server /' >> "$REP"
done

say ""
say "IF IT SCALES TO ZERO: each wake pays the start-up time before the first answer"
python3 - "$BOOT" "$RATE" <<'PY' | tee -a "$REP"
import sys
b, r = float(sys.argv[1]), float(sys.argv[2])
print(f"  {b:.0f} s x ${r:g}/h = ${b * r / 3600:.4f} per wake, before any request")
PY
say ""
say "WHAT THIS DOES NOT TELL YOU"
say "  The costs above assume the GPU is busy the whole time. A real server is"
say "  not: if it serves requests 30% of the day, each request really costs about"
say "  3.3x these numbers. Only your own traffic can show that share."
say ""
kill $SPID 2>/dev/null; sleep 5
[ "$OUT" = /workspace/planb ] && pkill -f "bin/vllm serve" 2>/dev/null
# CORPUS ROWS: the same numbers in the benchmarks/corpus/served-runs.csv format
python3 - "$OUT" "$OUT/corpus-rows.csv" <<'PY' >> "$OUT/run.out" 2>&1
import csv, re, sys
out_dir, dest = sys.argv[1], sys.argv[2]
rep = open(f"{out_dir}/report.txt").read()
def one(pat):
    m = re.search(pat, rep, re.M)
    return m.group(1).strip() if m else ""
gpu_line = [x.strip() for x in one(r"^gpu\s+(.*)$").split(",")]
gpu, mem, drv = (gpu_line + ["", "", ""])[:3]
flags = one(r"^flags\s+(.*)$")
lm = re.search(r"prompt ~(\d+) tokens, answer (\d+) tokens", rep)
cols = ["run_date","gpu","gpu_mem_mib","driver","host","usd_per_hour","engine","engine_version","model","extra_flags",
        "prompt_tokens_approx","output_tokens","ignore_eos","concurrency","window_s","requests_done","errors",
        "output_tok_per_s","latency_p50_s","latency_p95_s","usd_per_request","usd_per_1m_output_tokens",
        "meter_matches_server","startup_s","source"]
rows = []
for m in re.finditer(r"^\s+(\d+) at once: (\d+) done, (\d+) errors, ([\d,]+) output tok/s, latency p50 ([\d.]+) s / p95 ([\d.]+) s, \$([\d.]+) per request, \$([\d.]+) per 1M", rep, re.M):
    c = m.group(1)
    try:
        rc = open(f"{out_dir}/reconcile_{c}.txt").read()
    except OSError:
        rc = ""
    w = re.search(r"window\s+([\d.]+) s", rc)
    rows.append([one(r"^date_utc\s+(\d{4}-\d\d-\d\d)"), gpu, mem.replace(" MiB", ""), drv, "",
                 one(r"^price\s+\$([\d.]+)"), "vllm", one(r"^vllm\s+(\S+)"), one(r"^model\s+(\S+)"),
                 "" if flags == "none" else flags, lm.group(1) if lm else "", lm.group(2) if lm else "", "true",
                 c, w.group(1) if w else "", m.group(2), m.group(3), m.group(4).replace(",", ""), m.group(5), m.group(6),
                 m.group(7), m.group(8), "yes" if "Meter and server agree exactly." in rc else "no",
                 one(r"^start-up\s+([\d.]+) s"), ""])
with open(dest, "w", newline="") as f:
    wr = csv.writer(f, lineterminator="\n"); wr.writerow(cols); wr.writerows(rows)
print(f"{len(rows)} corpus rows -> {dest}")
PY
cp "$REP" /vllm-workspace/report-planb.txt 2>/dev/null
cp "$OUT/corpus-rows.csv" /vllm-workspace/corpus-rows.csv 2>/dev/null
tar czf /vllm-workspace/planb-files.tgz -C "$OUT" . 2>/dev/null
[ -d /vllm-workspace ] || tar czf "$OUT/planb-files.tgz" -C "$OUT" --exclude=planb-files.tgz . 2>/dev/null
say "Files: $REP, $OUT/corpus-rows.csv, planb-files.tgz (in /vllm-workspace if it exists, else in $OUT)"
say "Add your numbers to the public corpus (optional): open"
say "  https://github.com/qaisermehdi3-coder/qvunex/issues/new?template=add-gpu-numbers.yml"
say "  and paste corpus-rows.csv. Fill in host and price source."
say "ALL DONE. Download report-planb.txt, corpus-rows.csv and planb-files.tgz. On a rented GPU, DESTROY the instance."
