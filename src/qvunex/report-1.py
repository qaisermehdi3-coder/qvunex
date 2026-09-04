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

from .prices import cost_of, load as load_prices
from .schema import CALL, FALLBACK, GPU, RETRY, SESSION, TOKEN_FIELDS
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


def _quantile(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def _tasks(calls, cards):
    """Group calls into the units of work they belonged to.

    This is the number the whole thing exists to produce. A monthly bill divided
    by a request count is an average over a population nobody chose; "what did
    one finished outreach email cost, and what did the expensive ones have in
    common" is a question a person can act on. It can only be answered if the
    task id was attached at call time, which is what `qvunex.task()` does.

    Calls with no task id are not silently dropped and not spread across the
    tasks either. They are counted apart, because attributing them by proportion
    would be a guess dressed as a measurement.
    """
    by_id = {}
    loose = {"calls": 0, "cost_usd": 0.0, "unpriced": 0,
             "tokens": defaultdict(int)}

    for c in calls:
        cost = cost_of(c, cards)
        tid = c.get("task_id")
        if not tid:
            loose["calls"] += 1
            if cost is None:
                loose["unpriced"] += 1
            else:
                loose["cost_usd"] += cost
            for f in TOKEN_FIELDS:
                if c.get(f):
                    loose["tokens"][f] += c[f]
            continue

        t = by_id.get(tid)
        if t is None:
            t = by_id[tid] = {
                "task_id": tid,
                "task": c.get("task") or "(unnamed)",
                "calls": 0, "cost_usd": 0.0, "unpriced": 0,
                "wasted_usd": 0.0, "retries": 0, "fallbacks": 0, "errors": 0,
                "models": set(), "tokens": defaultdict(int),
                "first_ts": c["ts"], "last_end": c["ts"],
                "busy_ms": 0.0,
            }
        t["calls"] += 1
        t["busy_ms"] += float(c.get("dur_ms", 0.0))
        t["first_ts"] = min(t["first_ts"], c["ts"])
        t["last_end"] = max(t["last_end"], c["ts"] + float(c.get("dur_ms", 0)) / 1000.0)
        if c.get("model"):
            t["models"].add(c["model"])
        for f in TOKEN_FIELDS:
            if c.get(f):
                t["tokens"][f] += c[f]
        if cost is None:
            t["unpriced"] += 1
        else:
            t["cost_usd"] += cost
        status = c.get("status")
        if status == RETRY:
            t["retries"] += 1
            t["wasted_usd"] += cost or 0.0
        elif status == FALLBACK:
            t["fallbacks"] += 1
        if not c.get("ok", True):
            t["errors"] += 1

    tasks = []
    for t in by_id.values():
        t["models"] = sorted(t["models"])
        t["tokens"] = dict(t["tokens"])
        t["priced"] = t["unpriced"] == 0
        t["wall_s"] = max(t["last_end"] - t["first_ts"], 0.0)
        tasks.append(t)
    tasks.sort(key=lambda t: -t["cost_usd"])

    # --- per task *name*: the distribution, not just the mean ---------------
    by_name = defaultdict(list)
    for t in tasks:
        by_name[t["task"]].append(t)

    kinds = []
    for name, group in by_name.items():
        priced = [t for t in group if t["priced"]]
        costs = [t["cost_usd"] for t in priced]
        tokens = defaultdict(int)
        for t in group:
            for f, n in t["tokens"].items():
                tokens[f] += n
        kinds.append({
            "task": name,
            "n": len(group),
            "n_priced": len(priced),
            "total_usd": sum(costs) if costs else None,
            "mean_usd": statistics.mean(costs) if costs else None,
            "p50_usd": _quantile(costs, 0.50),
            "p95_usd": _quantile(costs, 0.95),
            "max_usd": max(costs) if costs else None,
            "mean_calls": statistics.mean([t["calls"] for t in group]),
            "mean_wall_s": statistics.mean([t["wall_s"] for t in group]),
            "retries": sum(t["retries"] for t in group),
            "fallbacks": sum(t["fallbacks"] for t in group),
            "errors": sum(t["errors"] for t in group),
            "wasted_usd": sum(t["wasted_usd"] for t in priced) if priced else None,
            "tokens": dict(tokens),
        })
    kinds.sort(key=lambda k: -(k["total_usd"] or 0))

    loose["tokens"] = dict(loose["tokens"])
    return tasks, kinds, loose


def analyse(path, rate_usd_hour=None, idle_threshold=IDLE_UTIL_THRESHOLD,
            prices=None):
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

    # ---- tokens and tasks ------------------------------------------------
    cards = prices if isinstance(prices, dict) else load_prices(prices)
    tasks, kinds, loose = _tasks(calls, cards)
    tokens_total = defaultdict(int)
    for c in calls:
        for f in TOKEN_FIELDS:
            if c.get(f):
                tokens_total[f] += c[f]
    token_cost = sum(t["cost_usd"] for t in tasks) + loose["cost_usd"]
    unpriced_calls = sum(t["unpriced"] for t in tasks) + loose["unpriced"]

    # ---- per endpoint ---------------------------------------------------
    by_ep = defaultdict(lambda: {"calls": 0, "inferences": 0,
                                 "busy_ms": 0.0, "lat": [], "errors": 0,
                                 "batches": [], "token_usd": 0.0,
                                 "unpriced": 0, "tokens": defaultdict(int)})
    for c in calls:
        e = by_ep[c["endpoint"]]
        e["calls"] += 1
        e["inferences"] += int(c.get("batch", 1))
        e["busy_ms"] += float(c.get("dur_ms", 0.0))
        e["lat"].append(float(c.get("dur_ms", 0.0)))
        e["batches"].append(int(c.get("batch", 1)))
        for f in TOKEN_FIELDS:
            if c.get(f):
                e["tokens"][f] += c[f]
        ec = cost_of(c, cards)
        if ec is None:
            e["unpriced"] += 1
        else:
            e["token_usd"] += ec
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
            "token_usd": e["token_usd"] if e["unpriced"] < e["calls"] else None,
            "unpriced": e["unpriced"],
            "tokens": dict(e["tokens"]),
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
        "tokens": dict(tokens_total),
        "token_cost_usd": token_cost if cards else None,
        "unpriced_calls": unpriced_calls,
        "priced_models": sorted(cards) if cards else [],
        "tasks": tasks,
        "task_kinds": kinds,
        "untasked": loose,
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

    # --- tokens ---
    tk = a.get("tokens") or {}
    if tk:
        add(bar)
        add("  TOKENS")
        add(bar)
        billed_in = tk.get("tokens_in", 0)
        cr, cw = tk.get("cache_read", 0), tk.get("cache_write", 0)
        add(f"  input (base rate)   {billed_in:>12,}")
        if cw or cr:
            add(f"  cache writes        {cw:>12,}   charged above the input rate")
            add(f"  cache reads         {cr:>12,}   charged at a tenth of it")
            offered = billed_in + cw + cr
            if offered:
                add(f"  cache hit rate      {_pct(cr / offered):>12}"
                    "   of all prompt tokens sent")
        add(f"  output              {tk.get('tokens_out', 0):>12,}")
        if tk.get("tokens_reasoning"):
            add(f"    of which thinking {tk['tokens_reasoning']:>12,}"
                "   billed inside output, not extra")
        if a.get("token_cost_usd") is not None:
            add(f"  token spend         {_usd(a['token_cost_usd']):>12}")
        if not a.get("priced_models"):
            add("  no rate card loaded -- tokens counted, not costed.")
            add("  Put one at ~/.qvunex/prices.txt or pass --prices FILE.")
        elif a.get("unpriced_calls"):
            add(f"  unpriced calls      {a['unpriced_calls']:>12,}"
                "   no rate card entry; excluded, not guessed")
        add("")

    # --- the headline: what one finished task cost ---
    kinds = a.get("task_kinds") or []
    if kinds:
        add(bar)
        add("  COST PER FINISHED TASK")
        add(bar)
        add(f"  {'task':<20}{'n':>5}{'mean':>11}{'p50':>11}{'p95':>11}{'calls':>8}")
        for k in kinds:
            add(f"  {k['task'][:20]:<20}{k['n']:>5}"
                f"{_usd(k['mean_usd']):>11}{_usd(k['p50_usd']):>11}"
                f"{_usd(k['p95_usd']):>11}{k['mean_calls']:>8.1f}")
        for k in kinds:
            if k["n_priced"] < k["n"]:
                add(f"  note: {k['n'] - k['n_priced']} of {k['n']} "
                    f"'{k['task'][:20]}' tasks contained a call with no price "
                    "and are left out of the figures above.")
        worst = [k for k in kinds if k["mean_usd"] and k["p95_usd"]
                 and k["p95_usd"] > k["mean_usd"] * 2]
        for k in worst:
            add(f"  spread: the dearest '{k['task'][:20]}' tasks cost "
                f"{k['p95_usd'] / k['mean_usd']:.1f}x the mean. An average "
                "hides that.")
        tax = sum(k["wasted_usd"] or 0 for k in kinds)
        rt = sum(k["retries"] for k in kinds)
        fb = sum(k["fallbacks"] for k in kinds)
        if rt or fb:
            add("")
            if rt:
                add(f"  retries             {rt:>5} call(s), {_usd(tax)}"
                    "   the same work paid for twice")
            if fb:
                add(f"  fallbacks           {fb:>5} call(s)"
                    "        a model you did not ask for answered")
        add("")

    loose = a.get("untasked") or {}
    if loose.get("calls") and kinds:
        add(f"  {loose['calls']:,} call(s) carried no task id"
            f" ({_usd(loose['cost_usd'])}). Real spend, unattributable.")
        add("  Wrap the work in `with qvunex.task(\"name\"):` to place it.")
        add("")

    # --- where the money goes ---
    add(bar)
    add("  COST BY ENDPOINT")
    add(bar)
    has_tok = any(e.get("token_usd") for e in a["endpoints"])
    if has_tok:
        # Ordered by what you pay, not by how long it ran. On an API the two
        # are barely related: the cheap step can hold the clock and the dear
        # one can return in a blink.
        rows = sorted(a["endpoints"], key=lambda e: -(e.get("token_usd") or 0))
        tot = sum(e.get("token_usd") or 0 for e in rows) or 1e-12
        add(f"  {'endpoint':<22}{'calls':>8}{'tokens $':>11}{'share':>8}{'per call':>11}")
        for e in rows:
            tu = e.get("token_usd")
            add(f"  {e['endpoint'][:22]:<22}{e['calls']:>8,}{_usd(tu):>11}"
                f"{(_pct(tu / tot) if tu is not None else 'n/a'):>8}"
                f"{(_usd(tu / e['calls']) if tu is not None else 'n/a'):>11}")
        if a.get("total_cost_usd") is not None:
            add("")
            add(f"  plus {_usd(a['total_cost_usd'])} of GPU time over the window,")
            add("  attributed by share of busy time:")
            for e in a["endpoints"]:
                add(f"  {e['endpoint'][:22]:<22}{e['inferences']:>9,}"
                    f"{_pct(e['share_of_busy']):>8}{_usd(e['cost_usd']):>11}")
    else:
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
    # Duty cycle, batch efficiency and idle burn are all questions about a
    # device you are renting by the hour. On a per-token API none of them mean
    # anything -- you are not paying for the gaps -- so the section is dropped
    # rather than printed full of numbers that look like findings.
    g = a["gpu"]
    if g or not a.get("tokens"):
        add(bar)
        add("  WASTE")
        add(bar)
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
    elif not a.get("tokens"):
        add("  no GPU samples -- nvidia-smi not available on this host")

    if g or not a.get("tokens"):
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
        priced = [e for e in a["endpoints"] if e.get("token_usd")]
        if priced:
            # When there are prices, rank by money. Busy time and spend point at
            # different endpoints often enough that using the wrong one sends
            # people to optimise the wrong step.
            tot = sum(e["token_usd"] for e in priced)
            top = max(priced, key=lambda e: e["token_usd"])
            if tot and top["token_usd"] / tot > 0.4:
                line = (f"'{top['endpoint'][:22]}' is "
                        f"{_pct(top['token_usd'] / tot)} of token spend")
                cheapest = min(priced, key=lambda e: e["token_usd"] / e["calls"])
                if cheapest is top and len(priced) > 1:
                    line += (f" while being the cheapest single call you make "
                             f"({_usd(top['token_usd'] / top['calls'])}). It is "
                             f"the {top['calls']:,} runs, not the price")
                out.append(line + ". Optimisation effort belongs there first.")
        elif a["endpoints"][0]["share_of_busy"] > 0.6:
            top = a["endpoints"][0]
            out.append(f"'{top['endpoint']}' is {_pct(top['share_of_busy'])} of all "
                       "compute. Optimisation effort belongs there first.")
    # Cache health is judged per endpoint, not over the whole corpus. One step
    # writing a cache nobody reads is invisible in the totals the moment some
    # other step has a healthy hit rate -- and the totals are what a dashboard
    # shows you.
    for e in a["endpoints"]:
        et = e.get("tokens") or {}
        cw, cr = et.get("cache_write", 0), et.get("cache_read", 0)
        if cw and not cr:
            out.append(f"'{e['endpoint'][:22]}' wrote {cw:,} tokens to the prompt "
                       "cache and read none back. A write costs more than not "
                       "caching at all, so that is a pure surcharge. Usually the "
                       "prefix is under the model's minimum (512-4096 tokens "
                       "depending on the model, below which the cache marker is "
                       "silently ignored), or something per-request sits above "
                       "the breakpoint.")
        elif cw and cr and cr < cw:
            out.append(f"'{e['endpoint'][:22]}' rebuilds its cache more often than "
                       f"it uses it ({cw:,} written, {cr:,} read). Check the TTL "
                       "against how far apart the requests actually arrive.")
    tk = a.get("tokens") or {}
    if tk.get("tokens_reasoning") and tk.get("tokens_out"):
        share = tk["tokens_reasoning"] / tk["tokens_out"]
        if share > 0.3:
            out.append(f"{_pct(share)} of output tokens are thinking tokens. They "
                       "are billed and they are invisible to any dashboard that "
                       "reads completion counts alone.")
    for k in (a.get("task_kinds") or []):
        if k["wasted_usd"] and k["total_usd"] and k["wasted_usd"] > k["total_usd"] * 0.05:
            out.append(f"Retries are {_pct(k['wasted_usd'] / k['total_usd'])} of "
                       f"what '{k['task'][:20]}' costs.")
    errs = sum(e["errors"] for e in a["endpoints"])
    if errs:
        out.append(f"{errs} call(s) raised. Failed inferences cost the same as "
                   "successful ones.")
    return out
