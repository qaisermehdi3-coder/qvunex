#!/usr/bin/env python3
"""Which measured GPU setup is cheapest per token at your answer-time target?

Reads served-runs.csv (same folder) and prints the measured setups whose
latency meets your target, cheapest per 1M output tokens first.

  python3 cheapest.py --p95 8
  python3 cheapest.py --p50 4 --price "A100=1.40" --price "3090=0.22"

Only measured points are used. Nothing is interpolated between load levels
or extrapolated to other GPUs, models or lengths. Costs assume the GPU is
busy the whole time; idle hours in a real day raise them.
Cost scales linearly with the hourly price. When a GPU was measured on several
hosts at different prices, all its rows are compared at one price: the lowest
measured for that GPU, or yours with --price (a row measured at another price
says so in its brackets).
Standard library only.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--p95", type=float, help="answer time target in seconds, 95th percentile")
    g.add_argument("--p50", type=float, help="answer time target in seconds, median")
    ap.add_argument("--model", help="only rows whose model contains this text")
    ap.add_argument("--price", action="append", default=[], metavar="GPU=USD_PER_HOUR",
                    help="re-price rows whose GPU name contains GPU, e.g. A100=1.40")
    ap.add_argument("--csv", default=os.path.join(HERE, "served-runs.csv"))
    a = ap.parse_args(argv)

    prices = []
    for p in a.price:
        if "=" not in p:
            ap.error(f"--price needs GPU=USD_PER_HOUR, got {p!r}")
        name, val = p.split("=", 1)
        prices.append((name.strip().lower(), float(val)))

    key, target = ("latency_p95_s", a.p95) if a.p95 is not None else ("latency_p50_s", a.p50)
    rows = list(csv.DictReader(open(a.csv, newline="")))
    if a.model:
        rows = [r for r in rows if a.model.lower() in r["model"].lower()]
    if not rows:
        print("no rows match", file=sys.stderr)
        return 1

    base = {}
    for r in rows:
        base[r["gpu"]] = min(base.get(r["gpu"], float("inf")), float(r["usd_per_hour"]))
    fits, too_slow = [], set()
    for r in rows:
        rate = base[r["gpu"]]
        for name, val in prices:
            if name in r["gpu"].lower():
                rate = val
        cost = float(r["usd_per_1m_output_tokens"]) * rate / float(r["usd_per_hour"])
        if float(r[key]) <= target:
            fits.append((cost, rate, r))
        else:
            too_slow.add(r["gpu"])

    label = key.split("_")[1]
    print(f"Target: {label} answer time <= {target:g} s   (measured points only, GPU busy the whole time)")
    print()
    if not fits:
        print("No measured setup meets this target.")
        return 0
    fits.sort(key=lambda x: x[0])
    best_per_gpu = {}
    for cost, rate, r in fits:
        best_per_gpu.setdefault(r["gpu"], (cost, rate, r))
    print("Cheapest measured setup per GPU that meets the target:")
    for gpu, (cost, rate, r) in sorted(best_per_gpu.items(), key=lambda kv: kv[1][0]):
        print(f"  ${cost:.3f} per 1M output tokens  {gpu} at ${rate:g}/h, "
              f"{r['concurrency']} at once, p50 {r['latency_p50_s']} s / p95 {r['latency_p95_s']} s, "
              f"{r['model']}, {r['engine']} {r['engine_version']}  [{r['source']}"
              + ("" if float(r["usd_per_hour"]) == rate else f", measured at ${float(r['usd_per_hour']):g}/h") + "]")
    never = sorted(g for g in too_slow if g not in best_per_gpu)
    if never:
        print()
        print("Measured but never fast enough at any level: " + ", ".join(never))
    if len(best_per_gpu) > 1:
        (c1, _, r1), (c2, _, r2) = sorted(best_per_gpu.values(), key=lambda x: x[0])[:2]
        print()
        if c2 / c1 < 1.05:
            print(f"{r1['gpu']} and {r2['gpu']} cost about the same per token at this target.")
        else:
            print(f"{r1['gpu']} is {c2 / c1:.1f}x cheaper per token than {r2['gpu']} at this target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
