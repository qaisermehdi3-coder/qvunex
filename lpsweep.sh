#!/usr/bin/env bash
# Qvunex budget sweep for vllm-project/vllm#54919. Prepared 27 Nov 2026.
# Runs longpair.sh once per setting on the same GPU (the model downloads once),
# then prints one summary line per setting.
#
#   curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/lpsweep.sh | bash -s -- RATE ["SETTINGS"]
#
# RATE is dollars per hour for the rented GPU. SETTINGS is a list of
# cache:budget pairs, default "off:512 off:1024 off:2048 off:4096 on:3136 on:4096":
#   off = no prefix caching, on = --enable-prefix-caching
#   the number = --max-num-batched-tokens
# Every run uses cyankiwi/Qwen3.8-27B-AWQ-INT4 and the other flags from the
# #54919 repro: --max-model-len 200000 --max-num-seqs 3
# --gpu-memory-utilization 0.95 --kv-cache-dtype fp8
#
# Why 3136: with prefix caching on, this hybrid model runs Mamba cache "align"
# mode and an attention block of 1568 tokens. vLLM 0.27.1's scheduler
# (_mamba_block_aligned_split) cuts prefill chunks to end on a block boundary,
# so a 4096 budget should give 3136-token chunks. on:3136 and on:4096 test that.
#
# When it prints ALL DONE, download report-sweep.txt and sweep-files.tgz from
# /vllm-workspace, then DESTROY the instance.

set -u
ROOT=${QVUNEX_SWEEP:-$HOME/qvunex-sweep}
mkdir -p "$ROOT" || { echo "cannot create $ROOT; set QVUNEX_SWEEP"; exit 1; }
export QVUNEX_SWEEP="$ROOT"
RAW=https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main

if [ "${SWEEP_INNER:-}" != "1" ]; then
  curl -sL "$RAW/lpsweep.sh" -o "$ROOT/lpsweep.sh"
  SWEEP_INNER=1 nohup bash "$ROOT/lpsweep.sh" "$@" > "$ROOT/sweep.out" 2>&1 &
  echo "started in background, pid $!  (report: $ROOT/report.txt)"
  sleep 3
  exec tail -n +1 -F "$ROOT/report.txt" 2>/dev/null
fi

RATE=${1:?usage: lpsweep.sh RATE ["SETTINGS"]}
SETTINGS=${2:-off:512 off:1024 off:2048 off:4096 on:3136 on:4096}
MODEL=cyankiwi/Qwen3.8-27B-AWQ-INT4
BASE="--max-model-len 200000 --max-num-seqs 3 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8"
REP=$ROOT/report.txt
: > "$REP"
say() { echo "$*" | tee -a "$REP"; }

# QVUNEX_LONGPAIR_SRC (a local file) is only for testing this script without GitHub
if [ -n "${QVUNEX_LONGPAIR_SRC:-}" ]; then cp "$QVUNEX_LONGPAIR_SRC" "$ROOT/longpair.sh"
else curl -sL "$RAW/longpair.sh" -o "$ROOT/longpair.sh"; fi

say "QVUNEX - TOKEN BUDGET SWEEP, TWO LONG PROMPTS AT ONCE (vllm#54919)"
say "date_utc   $(date -u +%FT%TZ)"
say "gpu        $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
say "cpu        $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1), nproc $(nproc)"
say "model      $MODEL"
say "base flags $BASE"
say "settings   $SETTINGS"
say "price      \$$RATE per hour for this GPU"
say "script     lpsweep.sh md5 $(md5sum "$ROOT/lpsweep.sh" 2>/dev/null | cut -c1-8), longpair.sh md5 $(md5sum "$ROOT/longpair.sh" | cut -c1-8)"
say ""
say "One line per setting, added as each one finishes (about 5 minutes each)."
say "A = the ~60k prompt, B = the ~75k prompt, sent together. 'A while B read' is A's"
say "decode speed between A's first token and B's first token."
say ""

cat > "$ROOT/line.py" <<'PY'
import json, os, re, sys
d, s = sys.argv[1], sys.argv[2]
def read(p):
    try:
        return open(os.path.join(d, p), errors="replace").read()
    except OSError:
        return ""
rep, log = read("report.txt"), read("server.log")
m = re.search(r"start-up\s+([\d.]+) s", rep)
startup = m.group(1) + " s" if m else "unknown"
m = re.search(r"max_num_batched_tokens=(\d+)", log)
logged = m.group(1) if m else "not in log"
m = re.search(r"Setting attention block size to (\d+)", log)
block = m.group(1) if m else "not in log"
align = "yes" if "Mamba cache mode is set to 'align'" in log else "no"
try:
    r = json.loads(read("result_longpair.json"))
    a, b = r["A"], r["B"]
except Exception:
    why = "server did not start" if "SERVER FAILED" in rep else "no result"
    print(f"  {s:<10} {why} (see {os.path.basename(d)}/report.txt); log budget {logged}")
    sys.exit()
f = lambda x, u="": "unknown" if x is None else f"{x}{u}"
print(f"  {s:<10} A while B read {f(a['rate_while_other_prefills'], ' tok/s')} over {f(a['window_s'], ' s')}, "
      f"A longest gap {f(a['max_gap'], ' s')}, first token A {f(a['first_token'], ' s')} / B {f(b['first_token'], ' s')}, "
      f"pair wall {r['wall']} s")
print(f"  {'':<10} log budget {logged}, align mode {align}, attention block {block}, start-up {startup}, "
      f"tokens A {a['tokens']} / B {b['tokens']}"
      + ("".join(f", error {k}: {x['error']}" for k, x in (('A', a), ('B', b)) if x.get('error'))))
PY

i=0
for s in $SETTINGS; do
  i=$((i + 1))
  pc=${s%%:*}; b=${s##*:}
  case "$pc" in
    on) extra=" --enable-prefix-caching" ;;
    off) extra="" ;;
    *) say "  $s: not a valid setting (use on:N or off:N), skipped"; continue ;;
  esac
  case "$b" in ''|*[!0-9]*) say "  $s: not a valid budget, skipped"; continue ;; esac
  d="$ROOT/run$i-$pc-$b"
  mkdir -p "$d"
  cp "$ROOT/longpair.sh" "$d/longpair.sh"
  QVUNEX_OUT="$d" LONGPAIR_INNER=1 bash "$d/longpair.sh" "$RATE" "$MODEL" "$BASE --max-num-batched-tokens $b$extra" > "$d/stdout.txt" 2>&1
  python3 "$ROOT/line.py" "$d" "$s" | tee -a "$REP"
done

say ""
say "WHAT THIS DOES NOT TELL YOU"
say "  One run per setting, one pair of prompts, synthetic text (a repeated word),"
say "  completions API with ignore_eos. One GPU and one model."
say ""
if [ -d /vllm-workspace ]; then
  rm -f /vllm-workspace/report-longpair.txt /vllm-workspace/longpair-files.tgz
  cp "$REP" /vllm-workspace/report-sweep.txt
  tar czf /tmp/sweep-files.tgz --warning=no-file-changed -C "$ROOT" . && mv /tmp/sweep-files.tgz /vllm-workspace/sweep-files.tgz
  say "Files: /vllm-workspace/report-sweep.txt and /vllm-workspace/sweep-files.tgz"
else
  tar czf /tmp/sweep-files.tgz --warning=no-file-changed -C "$ROOT" --exclude=sweep-files.tgz . && mv /tmp/sweep-files.tgz "$ROOT/sweep-files.tgz"
  say "Files: $REP and $ROOT/sweep-files.tgz"
fi
say "ALL DONE. Download report-sweep.txt and sweep-files.tgz. On a rented GPU, DESTROY the instance."
