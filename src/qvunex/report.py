"""Turn a corpus file into the six numbers a customer cannot currently get.

The whole product thesis is that a company running AI in production knows exactly
one number — the monthly bill — and cannot break it down by feature, endpoint or
prediction. This module is the smallest useful answer to that.

Honesty rules that shape the code (Blueprint Rule #4 — a meter's only asset is
trust, and one inflated number discovered ends it permanently):

* Costs are **attributed**, not measured. We know total GPU-hours and the share of
  busy time each endpoint accounts for; we divide. Under concurrency, summed call
  duration exceeds wall time, so `duty_cycle` is capped at 1.0 and flagged rather
  than silently normalised away.
* Anything that needs a price we do not have is reported as unavailable, never
  estimated from a default rate.
* The idle threshold is a stated choice, printed in the output, not a hidden
  constant.
"""

import statistics
from collections import defaultdict

from .schema import CALL, GPU, SESSION
from .store import Store

IDLE_UTIL_THRESHOLD = 5.0   # percent; a GPU below this is doing nothing useful


def _pct(x):
    return f"{x*100:.1f}%"


def _usd(x):
    if x is None:
        return "n/a"
    if x < 0.01:
        return f"${x:.4f}"
    return f"${x:,.2f}"


def analyse(path, rate_usd_hour=None, idle_threshold=IDLE_UTIL_THRESHOLD):
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
        return {"empty": True, "path": path}

    if rate_usd_hour is None:
        for s in reversed(sessions):
            r = (s.get("config") or {}).get("rate_usd_hour")
            if r:
                rate_usd_hour = float(r)
                break
    rate_usd_hour = float(rate_usd_hour or 0) or None

    ts = [c["ts"] for c in calls] + [g["ts"] for g in gpus]
    t0, t1 = min(ts), max(ts)
    wall_s = max(t1 - t0, 1e-9)

    n_gpus = 0
    for s in sessions:
        n_gpus = max(n_gpus, len(s.get("gpus") or []))
    if not n_gpus and gpus:
        n_gpus = len({g["index"] for g in gpus})

    total_cost = (rate_usd_hour * (wall_s / 3600.0) * max(n_gpus, 1)
                  if rate_usd_hour else None)

    # ---- per endpoint ---------------------------------------------------
    by_ep = defaultdict(lambda: {"calls": 0, "inferences": 0,
                                 "busy_ms": 0.0, "lat": [], "errors": 0,
                                 "batches": []})
    for c in calls:
        e = by_ep[c["endpoint"]]
        e["calls"] += 1
        e["inferences"] += int(c.get("batch", 1))
        e["busy_ms"] += float(c.get("dur_ms", 0.0))
        e["lat"].append(float(c.get("dur_ms", 0.0)))
        e["batches"].append(int(c.get("batch", 1)))
        if not c.get("ok", True):
            e["errors"] += 1

    total_busy_ms = sum(e["busy_ms"] for e in by_ep.values()) or 1e-9
    total_inferences = sum(e["inferences"] for e in by_ep.values())

    endpoints = []
    for name, e in sorted(by_ep.items(), key=lambda kv: -kv[1]["busy_ms"]):
        lat = sorted(e["lat"])
        share = e["busy_ms"] / total_busy_ms
        cost = total_cost * share if total_cost is not None else None
        endpoints.append({
            "endpoint": name,
            "calls": e["calls"],
            "inferences": e["inferences"],
            "errors": e["errors"],
            "share_of_busy": share,
            "cost_usd": cost,
            "cpki_usd": (cost / e["inferences"] * 1000
                         if cost is not None and e["inferences"] else None),
            "mean_batch": statistics.mean(e["batches"]),
            "max_batch": max(e["batches"]),
            "latency_ms": {
                "mean": statistics.mean(lat),
                "p50": lat[len(lat) // 2],
                "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
                "p99": lat[min(len(lat) - 1, int(len(lat) * 0.99))],
            },
        })

    # ---- utilisation & idle burn ---------------------------------------
    util = None
    if gpus:
        u = [g["util"] for g in gpus]
        idle_samples = sum(1 for x in u if x < idle_threshold)
        idle_frac = idle_samples / len(u)
        power = [g["power_w"] for g in gpus if g.get("power_w") is not None]
        util = {
            "samples": len(u),
            "mean_util": statistics.mean(u),
            "median_util": statistics.median(u),
            "idle_fraction": idle_frac,
            "idle_threshold": idle_threshold,
            "idle_cost_usd": total_cost * idle_frac if total_cost is not None else None,
            "mean_power_w": statistics.mean(power) if power else None,
            "peak_power_w": max(power) if power else None,
            "mean_mem_used_mib": statistics.mean(
                [g["mem_used_mib"] for g in gpus]),
            "mem_total_mib": max([g["mem_total_mib"] for g in gpus]),
        }

    duty = min(total_busy_ms / 1000.0 / wall_s, 1.0)
    concurrent = (total_busy_ms / 1000.0) > wall_s * 1.05

    all_batches = [b for e in by_ep.values() for b in e["batches"]]
    max_batch = max(all_batches)
    batch_eff = statistics.mean(all_batches) / max_batch if max_batch else 1.0

    return {
        "empty": False,
        "path": path,
        "window_s": wall_s,
        "n_gpus": n_gpus,
        "rate_usd_hour": rate_usd_hour,
        "total_cost_usd": total_cost,
        "total_calls": len(calls),
        "total_inferences": total_inferences,
        "cpki_usd": (total_cost / total_inferences * 1000
                     if total_cost is not None and total_inferences else None),
        "duty_cycle": duty,
        "concurrent": concurrent,
        "batch_efficiency": batch_eff,
        "max_batch": max_batch,
        "endpoints": endpoints,
        "gpu": util,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render(a, width=68):
    if a.get("empty"):
        return (f"No calls recorded in {a['path']}.\n"
                "Add @meter(\"name\") to an inference function and run it once.")

    L = []
    add = L.append
    bar = "-" * width

    add("=" * width)
    add("  QVUNEX METER REPORT")
    add("=" * width)
    add(f"  window        {a['window_s']:.1f} s")
    add(f"  calls         {a['total_calls']:,}")
    add(f"  inferences    {a['total_inferences']:,}")
    if a["rate_usd_hour"]:
        add(f"  gpu rate      ${a['rate_usd_hour']:.4f}/hr x {max(a['n_gpus'],1)} device(s)")
        add(f"  window cost   {_usd(a['total_cost_usd'])}")
        add(f"  CPKI          {_usd(a['cpki_usd'])} per 1,000 inferences")
    else:
        add("  gpu rate      not set -- costs unavailable")
        add("                qvunex.configure(rate_usd_hour=...) to enable")
    add("")

    # --- where the money goes ---
    add(bar)
    add("  COST BY ENDPOINT")
    add(bar)
    add(f"  {'endpoint':<22}{'inf':>9}{'share':>8}{'cost':>11}{'CPKI':>12}")
    for e in a["endpoints"]:
        add(f"  {e['endpoint'][:22]:<22}{e['inferences']:>9,}"
            f"{_pct(e['share_of_busy']):>8}"
            f"{_usd(e['cost_usd']):>11}{_usd(e['cpki_usd']):>12}")
    add("")

    # --- latency ---
    add(bar)
    add("  LATENCY (ms)")
    add(bar)
    add(f"  {'endpoint':<22}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}")
    for e in a["endpoints"]:
        l = e["latency_ms"]
        add(f"  {e['endpoint'][:22]:<22}{l['mean']:>9.1f}{l['p50']:>9.1f}"
            f"{l['p95']:>9.1f}{l['p99']:>9.1f}")
    add("")

    # --- waste ---
    add(bar)
    add("  WASTE")
    add(bar)
    g = a["gpu"]
    if g:
        add(f"  mean GPU utilisation      {g['mean_util']:.1f}%")
        add(f"  time below {g['idle_threshold']:.0f}% util        "
            f"{_pct(g['idle_fraction'])}")
        if g["idle_cost_usd"] is not None:
            add(f"  cost of that idle time    {_usd(g['idle_cost_usd'])}"
                "   <-- paid for, not used")
        if g["mean_power_w"]:
            add(f"  mean board power          {g['mean_power_w']:.1f} W"
                f"  (peak {g['peak_power_w']:.1f} W)")
        add(f"  memory in use             {g['mean_mem_used_mib']:.0f} /"
            f" {g['mem_total_mib']:.0f} MiB"
            f"  ({_pct(g['mean_mem_used_mib']/max(g['mem_total_mib'],1))})")
    else:
        add("  no GPU samples -- nvidia-smi not available on this host")

    add(f"  duty cycle                {_pct(a['duty_cycle'])}"
        "   time actually inside inference")
    add(f"  batch efficiency          {_pct(a['batch_efficiency'])}"
        f"   mean batch vs max seen ({a['max_batch']})")
    if a["concurrent"]:
        add("")
        add("  note: summed call time exceeds wall time -- calls overlap.")
        add("        duty cycle is capped at 100% and cost shares are")
        add("        proportional, not causal.")
    add("")

    # --- what to do ---
    add(bar)
    add("  WHAT THIS SUGGESTS")
    add(bar)
    sug = _suggestions(a)
    if sug:
        for s in sug:
            add(f"  * {s}")
    else:
        add("  Nothing obviously wasteful in this window.")
    add("")
    add(bar)
    add("  Measured locally. Nothing left this machine.")
    add("  Run `qvunex checklist` before publishing any of these numbers.")
    add(bar)
    return "\n".join(L)


def _suggestions(a):
    """Only fire on thresholds that are defensible. A meter that cries wolf
    stops being read, and the suggestion is worth less than the trust."""
    out = []
    g = a["gpu"]
    if g and g["idle_fraction"] > 0.30:
        msg = (f"GPU sat below {g['idle_threshold']:.0f}% utilisation for "
               f"{_pct(g['idle_fraction'])} of the window")
        if g["idle_cost_usd"]:
            msg += f" ({_usd(g['idle_cost_usd'])})"
        out.append(msg + ". Look at autoscaling or consolidating onto fewer devices.")
    if g and g["mean_util"] < 40 and not (g["idle_fraction"] > 0.30):
        out.append(f"Mean utilisation {g['mean_util']:.0f}% -- the device is busy but "
                   "not saturated. Larger batches or co-locating a second workload "
                   "would use what you are already paying for.")
    if a["batch_efficiency"] < 0.6 and a["max_batch"] > 1:
        out.append(f"Mean batch is {_pct(a['batch_efficiency'])} of the largest batch "
                   f"seen ({a['max_batch']}). Padding or partial batches are costing "
                   "throughput.")
    if g and g["mem_total_mib"] and \
       g["mean_mem_used_mib"] / g["mem_total_mib"] < 0.25:
        out.append(f"Only {_pct(g['mean_mem_used_mib']/g['mem_total_mib'])} of GPU "
                   "memory is in use -- a smaller or cheaper device may fit this "
                   "workload.")
    if len(a["endpoints"]) > 1:
        top = a["endpoints"][0]
        if top["share_of_busy"] > 0.6:
            out.append(f"'{top['endpoint']}' is {_pct(top['share_of_busy'])} of all "
                       "compute. Optimisation effort belongs there first.")
    errs = sum(e["errors"] for e in a["endpoints"])
    if errs:
        out.append(f"{errs} call(s) raised. Failed inferences cost the same as "
                   "successful ones.")
    return out
