"""Emit a filled-in Comparability Checklist from a corpus file.

The checklist (https://gist.github.com/qaisermehdi3-coder/b00f296641681695daf90e5a500d0d23)
lists what an inference measurement must state before it can be compared to anyone
else's. A document nobody fills in is not a standard — so the meter fills in what it
can observe and marks the rest, loudly.

Three states per field:

  observed   the meter measured it. Not editable, not guessable.
  declared   the caller passed it to qvunex.configure(context={...}).
  MISSING    nobody supplied it. Printed in full so it cannot be quietly skipped.

The MISSING lines are the point. A checklist that silently omits what it does not
know would be worse than no checklist, because it would look complete.
"""

import statistics
from collections import defaultdict

from .schema import CALL, GPU, SESSION
from .store import Store

CHECKLIST_URL = ("https://gist.github.com/qaisermehdi3-coder/"
                 "b00f296641681695daf90e5a500d0d23")

# field id -> (label, key the caller can declare it under)
DECLARABLE = [
    ("1.2", "gpu_count / parallelism",      "parallelism"),
    ("1.3", "attention_backend",            "attention_backend"),
    ("2.1", "engine / version",             "engine"),
    ("2.2", "cuda_graphs_enabled",          "cuda_graphs"),
    ("2.3", "batching_mode",                "batching_mode"),
    ("2.4", "max_model_len / kv_cache",     "max_model_len"),
    ("3.1", "model_id / revision",          "model_id"),
    ("3.2", "weight_format / compute_dtype", "dtype"),
    ("3.3", "accuracy_check",               "accuracy_check"),
    ("4.2", "input_tokens / output_tokens", "tokens"),
    ("4.3", "eos_handling",                 "eos_handling"),
    ("4.4", "prompt_diversity",             "prompt_diversity"),
    ("5.1", "warmup_performed",             "warmup"),
    ("5.3", "power_method",                 "power_method"),
    ("6.1", "hourly_rate source",           "rate_source"),
    ("6.2", "idle_time_included",           "idle_included"),
]


def _observed(path):
    """Everything the meter can state without being told."""
    calls, gpus, sessions = [], [], []
    for r in Store.read(path):
        t = r.get("t")
        if t == CALL:
            calls.append(r)
        elif t == GPU:
            gpus.append(r)
        elif t == SESSION:
            sessions.append(r)

    if not calls:
        return None

    out = {}

    # 1.1 hardware
    devices = []
    for s in sessions:
        for g in (s.get("gpus") or []):
            devices.append(g)
    if devices:
        names = sorted({d["name"] for d in devices})
        out["1.1"] = ", ".join(names)
        limits = sorted({d.get("power_limit_w") for d in devices if d.get("power_limit_w")})
        if limits:
            out["1.1"] += f"  (power limit {'/'.join(f'{x:.0f}W' for x in limits)})"
        drivers = sorted({d.get("driver") for d in devices if d.get("driver")})
        if drivers:
            out["1.1"] += f", driver {'/'.join(drivers)}"
    elif gpus:
        out["1.1"] = f"{len({g['index'] for g in gpus})} CUDA device(s), model not captured"

    # 4.1 batch
    batches = [int(c.get("batch", 1)) for c in calls]
    uniq = sorted(set(batches))
    if len(uniq) == 1:
        out["4.1"] = f"batch {uniq[0]}, fixed"
    else:
        out["4.1"] = (f"batch {min(uniq)}–{max(uniq)}, "
                      f"mean {statistics.mean(batches):.1f} "
                      f"({len(uniq)} distinct sizes)")

    # 5.2 repeats and spread
    #
    # Spread must be computed WITHIN a fixed (endpoint, batch) group. Pooling
    # across batch sizes measures the batch mix, not reproducibility, and would
    # report an alarming number that means nothing — the exact failure this
    # checklist exists to catch.
    by_ep = defaultdict(list)
    groups = defaultdict(list)
    for c in calls:
        by_ep[c["endpoint"]].append(c)
        groups[(c["endpoint"], int(c.get("batch", 1)))].append(
            float(c.get("dur_ms", 0.0)))

    spreads = [statistics.stdev(v) / statistics.mean(v)
               for v in groups.values()
               if len(v) > 2 and statistics.mean(v) > 0]

    line = f"{len(calls):,} calls across {len(by_ep)} endpoint(s)"
    if spreads:
        line += (f"; latency CV within a fixed batch size "
                 f"{min(spreads)*100:.0f}–{max(spreads)*100:.0f}% "
                 f"over {len(spreads)} group(s)")
    else:
        line += "; too few repeats at any single batch size to state a spread"
    line += (". NOTE: single session — this is not run-to-run reproducibility. "
             "Repeat on a fresh session before publishing.")
    out["5.2"] = line

    # 5.4 instrumentation
    out["5.4"] = ("qvunex decorator, out-of-band background sampler; "
                  "no per-layer hooks, so timing is not synchronisation-distorted")

    # 6.1 / 6.3 cost
    rate = None
    for s in reversed(sessions):
        r = (s.get("config") or {}).get("rate_usd_hour")
        if r:
            rate = float(r)
            break
    out["6.1"] = f"${rate:.4f}/hr (assumed)" if rate else None
    out["6.3"] = ("cost per 1,000 inferences (CPKI) and per endpoint; "
                  "see 4.2 before converting to per-token")

    return out


def build(path, context=None):
    """Return (lines, n_missing)."""
    obs = _observed(path)
    if obs is None:
        return ([f"No calls recorded in {path} — nothing to report."], 0)

    ctx = dict(context or {})
    L, missing = [], 0
    W = 74

    L.append("=" * W)
    L.append("  COMPARABILITY CHECKLIST v0.1  —  auto-filled by qvunex")
    L.append("  " + CHECKLIST_URL)
    L.append("=" * W)
    L.append("")

    groups = [
        ("01  HARDWARE",      ["1.1", "1.2", "1.3"]),
        ("02  SERVING STACK", ["2.1", "2.2", "2.3", "2.4"]),
        ("03  MODEL",         ["3.1", "3.2", "3.3"]),
        ("04  WORKLOAD",      ["4.1", "4.2", "4.3", "4.4"]),
        ("05  MEASUREMENT",   ["5.1", "5.2", "5.3", "5.4"]),
        ("06  COST BASIS",    ["6.1", "6.2", "6.3"]),
    ]
    labels = {fid: lab for fid, lab, _ in DECLARABLE}
    labels.update({
        "1.1": "gpu_model", "4.1": "batch_size / concurrency",
        "5.2": "repeat_count / variance", "5.4": "instrumentation_overhead",
        "6.3": "unit",
    })
    keys = {fid: key for fid, _, key in DECLARABLE}

    for title, fids in groups:
        L.append(title)
        L.append("-" * W)
        for fid in fids:
            label = labels.get(fid, fid)
            val = obs.get(fid)
            tag = "observed"
            if val is None:
                key = keys.get(fid)
                if key and ctx.get(key) not in (None, ""):
                    val, tag = str(ctx[key]), "declared"
            if val is None:
                missing += 1
                L.append(f"  {fid}  {label}")
                L.append(f"        >> MISSING — qvunex cannot observe this. State it yourself.")
            else:
                L.append(f"  {fid}  {label}   [{tag}]")
                for i, chunk in enumerate(_wrap(str(val), W - 10)):
                    L.append(f"        {chunk}")
        L.append("")

    L.append("=" * W)
    if missing:
        L.append(f"  {missing} field(s) MISSING. A number published without them is not")
        L.append("  comparable to anyone else's. Declare them like this:")
        L.append("")
        L.append("      qvunex.configure(rate_usd_hour=0.35, context={")
        L.append('          \"engine\": \"vLLM 0.27.1\",')
        L.append('          \"cuda_graphs\": True,')
        L.append('          \"model_id\": \"Qwen/Qwen2.5-1.5B-Instruct\",')
        L.append('          \"dtype\": \"fp16 weights, fp16 compute\",')
        L.append("      })")
    else:
        L.append("  All fields supplied. This measurement is comparable.")
    L.append("=" * W)
    return (L, missing)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out or [""]


def render(path, context=None):
    lines, _ = build(path, context)
    return "\n".join(lines)
