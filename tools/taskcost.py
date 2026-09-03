#!/usr/bin/env python3
"""
taskcost.py - what does one finished task cost, and which step is actually eating it.

Provider dashboards aggregate by model and by key. They almost never aggregate by
task, so when the number moves you can see that spend went up and not which job
did it. This takes a description of your pipeline and answers the other question,
without you shipping any traffic data anywhere.

It answers three things:

  1. Cost per FINISHED TASK, with every step ranked by its share.
     A step is counted as many times as it runs. A cheap call that fires once per
     item usually beats an expensive call that fires once per task, and nobody
     sees that until it is ranked.

  2. The same cost in three cache states: cold, typical, warm.
     Cost is not a number, it is a distribution over cache states. A bill that
     looks wrong is often just a different mix of the three.

  3. Whether your caching is silently doing nothing.
     Prompt caching has a minimum prompt length that differs per model. Below it,
     cache_control is ignored with no error and you quietly pay full price. And a
     cache write only happens at your breakpoint, so if anything inside that
     breakpoint changes per request the prefix never matches and you pay write
     prices forever while getting zero reads.

Usage:

    python taskcost.py pipeline.txt

Input format (plain text, see example.txt):

    task: outbound email drafting
    hit_rate: 0.5

    price: claude-sonnet-4.6
      input_per_mtok: 3.00
      output_per_mtok: 15.00

    stage: draft the email
      model: claude-sonnet-4.6
      input_tokens: 3200
      output_tokens: 600
      runs_per_task: 1
      cached_prefix_tokens: 2400
      cache_ttl: 5m
      breakpoint_stable: yes

Prices are yours to supply, deliberately. Rate cards change and a tool that ships
stale prices is worse than one that asks. The cache multipliers and per-model
minimums below are from Anthropic's prompt caching documentation - check them
against the current page before you trust a number.

MIT licensed. Part of qvunex: github.com/qaisermehdi3-coder/qvunex
"""

import re
import sys

# Cache write costs MORE than not caching at all. A cold run is not "no discount".
WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}
DEFAULT_READ_MULTIPLIER = 0.1

# Minimum cacheable prompt length. Below this, cache_control is ignored silently.
CACHE_MIN_TOKENS = {
    512:  ["fable-5.1", "mythos-5.1", "opus-5", "fable-5", "mythos-5"],
    1024: ["opus-4.8", "sonnet-5", "sonnet-4.6", "sonnet-4.5",
           "opus-4.1", "opus-4", "sonnet-4"],
    2048: ["mythos-preview", "opus-4.7", "haiku-3.5"],
    4096: ["opus-4.6", "opus-4.5", "haiku-4.5"],
}

# These two read cache at a lower rate than everything else.
LOW_READ_MULTIPLIER_MODELS = ["fable-5.1", "mythos-5.1"]


def normalise(name):
    n = name.strip().lower()
    n = n.replace("claude-", "").replace("claude ", "")
    n = n.replace(" ", "-").replace("_", "-")
    return n


def lookup_cache_min(model):
    n = normalise(model)
    for minimum, names in CACHE_MIN_TOKENS.items():
        for known in names:
            if n == known or n.startswith(known):
                return minimum, True
    return 1024, False          # a guess, and we say so


def lookup_read_multiplier(model):
    n = normalise(model)
    for known in LOW_READ_MULTIPLIER_MODELS:
        if n == known or n.startswith(known):
            return 0.025
    return DEFAULT_READ_MULTIPLIER


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------

TRUE = ("yes", "true", "y", "1")
FALSE = ("no", "false", "n", "0")


def parse(text):
    doc = {"task": "untitled task", "hit_rate": 0.5, "prices": {}, "stages": []}
    current = None

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError("cannot read this line, expected 'key: value':\n  %s"
                             % raw.strip())

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "task":
            doc["task"] = value
            current = None
        elif key == "hit_rate":
            doc["hit_rate"] = float(value)
            current = None
        elif key == "price":
            current = {"_kind": "price", "model": value}
            doc["prices"][normalise(value)] = current
        elif key == "stage":
            current = {"_kind": "stage", "name": value}
            doc["stages"].append(current)
        else:
            if current is None:
                raise ValueError("'%s' appears before any 'stage:' or 'price:'"
                                 % key)
            current[key] = value

    if not doc["stages"]:
        raise ValueError("no stages found")
    return doc


def num(block, key, default=None, cast=float):
    if key not in block:
        if default is None:
            raise ValueError("stage '%s' is missing '%s'"
                             % (block.get("name", block.get("model")), key))
        return default
    return cast(str(block[key]).replace(",", "").replace("$", ""))


def flag(block, key, default=True):
    if key not in block:
        return default
    v = str(block[key]).strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError("'%s' should be yes or no, got %r" % (key, block[key]))


# ----------------------------------------------------------------------------
# costing
# ----------------------------------------------------------------------------

def cost_stage(stage, prices):
    """Return per-task costs in each cache state, plus any warnings."""
    name = stage.get("name", "unnamed")
    model = stage.get("model")
    if not model:
        raise ValueError("stage '%s' has no model" % name)

    price = prices.get(normalise(model))
    if price is None:
        raise ValueError("no price block for model '%s' - add one" % model)

    p_in = num(price, "input_per_mtok") / 1_000_000.0
    p_out = num(price, "output_per_mtok") / 1_000_000.0

    total_in = num(stage, "input_tokens")
    out = num(stage, "output_tokens")
    runs = num(stage, "runs_per_task", 1.0)
    cached = num(stage, "cached_prefix_tokens", 0.0)
    ttl = str(stage.get("cache_ttl", "5m")).strip().lower()
    stable = flag(stage, "breakpoint_stable", True)

    if ttl not in WRITE_MULTIPLIER:
        raise ValueError("stage '%s': cache_ttl should be 5m or 1h" % name)
    if cached > total_in:
        raise ValueError("stage '%s': cached_prefix_tokens is larger than "
                         "input_tokens" % name)

    warnings = []

    # per-model overrides win over the built-in table
    if "cache_min_tokens" in price:
        cache_min, known = num(price, "cache_min_tokens", cast=int), True
    else:
        cache_min, known = lookup_cache_min(model)
        if not known and cached > 0:
            warnings.append(
                "model '%s' is not in the built-in minimum table, assuming %d. "
                "Set cache_min_tokens in its price block to be sure."
                % (model, cache_min))

    if "cache_read_multiplier" in price:
        read_mult = num(price, "cache_read_multiplier")
    else:
        read_mult = lookup_read_multiplier(model)
    write_mult = WRITE_MULTIPLIER[ttl]

    uncached = total_in - cached
    base_out = out * p_out

    cache_engages = cached > 0 and cached >= cache_min

    if cached > 0 and not cache_engages:
        warnings.append(
            "cached prefix is %d tokens, below the %d minimum for this model. "
            "cache_control is ignored with no error - you are paying full price "
            "in every state and a warm run is not cheaper."
            % (int(cached), cache_min))

    if cache_engages and not stable:
        warnings.append(
            "breakpoint_stable is no, so the prefix hash never matches. Every "
            "call pays the %.2fx write price and you get zero reads. This is what "
            "a changed emoji or a version bump inside the breakpoint does."
            % write_mult)

    if not cache_engages:
        # caching does nothing, all three states are identical
        one_call_cold = one_call_warm = total_in * p_in + base_out
    elif not stable:
        one_call_cold = cached * p_in * write_mult + uncached * p_in + base_out
        one_call_warm = one_call_cold          # never reads
    else:
        one_call_cold = cached * p_in * write_mult + uncached * p_in + base_out
        one_call_warm = cached * p_in * read_mult + uncached * p_in + base_out

    if cached > 0 and ttl == "1h" and cache_engages:
        warnings.append(
            "1h TTL costs 2x on every write. Worth it only if the prefix is "
            "re-read often inside the hour - otherwise 5m is cheaper.")

    return {
        "name": name,
        "model": model,
        "runs": runs,
        "cold": one_call_cold * runs,
        "warm": one_call_warm * runs,
        "cache_engages": cache_engages,
        "stable": stable,
        "cached_tokens": cached,
        "cache_min": cache_min,
        "read_mult": read_mult,
        "write_mult": write_mult,
        "uncached_in": uncached,
        "p_in": p_in,
        "base_out": base_out,
        "warnings": warnings,
    }


def money(x):
    if x >= 1:
        return "$%.2f" % x
    if x >= 0.01:
        return "$%.4f" % x
    return "$%.6f" % x


def report(doc):
    hit = doc["hit_rate"]
    if not 0.0 <= hit <= 1.0:
        raise ValueError("hit_rate should be between 0 and 1")

    rows = [cost_stage(s, doc["prices"]) for s in doc["stages"]]
    for r in rows:
        r["typical"] = hit * r["warm"] + (1 - hit) * r["cold"]

    cold = sum(r["cold"] for r in rows)
    warm = sum(r["warm"] for r in rows)
    typical = sum(r["typical"] for r in rows)

    line = "=" * 74
    print(line)
    print("  COST PER FINISHED TASK  -  %s" % doc["task"])
    print(line)
    print("  stages        %d" % len(rows))
    print("  hit_rate      %.0f%%  (used for the typical column)" % (hit * 100))
    print("")
    print("  cold    (nothing cached yet)      %s per task" % money(cold))
    print("  typical (%.0f%% of prefixes hit)    %s per task"
          % (hit * 100, money(typical)))
    print("  warm    (everything cached)       %s per task" % money(warm))
    if warm > 0:
        print("")
        print("  cold is %.2fx warm. A bill that looks wrong is often just a"
              % (cold / warm))
        print("  different mix of these three, not a regression.")

    print("")
    print("-" * 74)
    print("  WHERE THE MONEY GOES  (typical state, ranked)")
    print("-" * 74)
    print("  %-26s %-20s %7s %10s %7s"
          % ("stage", "model", "runs", "per task", "share"))
    for r in sorted(rows, key=lambda x: -x["typical"]):
        share = (r["typical"] / typical * 100) if typical else 0
        print("  %-26s %-20s %7s %10s %6.1f%%"
              % (r["name"][:26], r["model"][:20],
                 ("%g" % r["runs"]), money(r["typical"]), share))

    top = max(rows, key=lambda x: x["typical"])
    top_share = (top["typical"] / typical * 100) if typical else 0
    if top_share >= 40:
        print("")
        print("  '%s' is %.0f%% of the task." % (top["name"], top_share))
        if top["runs"] > 1:
            print("  It is not the dearest call - it is a %g-times-per-task call."
                  % top["runs"])
            print("  Per call it costs %s." % money(top["typical"] / top["runs"]))

    problems = [(r, w) for r in rows for w in r["warnings"]]
    print("")
    print("-" * 74)
    if problems:
        print("  WARNINGS  (%d)" % len(problems))
        print("-" * 74)
        for r, w in problems:
            print("  [%s]" % r["name"])
            for chunk in wrap(w, 68):
                print("    %s" % chunk)
            print("")
    else:
        print("  WARNINGS")
        print("-" * 74)
        print("  none. Every cached prefix clears its model's minimum and every")
        print("  breakpoint is stable.")
        print("")

    dead = [r for r in rows if r["cached_tokens"] > 0 and not r["cache_engages"]]
    if dead:
        print("-" * 74)
        print("  WHAT FIXING THE DEAD CACHES WOULD BE WORTH")
        print("-" * 74)
        saved = 0.0
        for r in dead:
            would_warm = (r["cached_tokens"] * r["p_in"] * r["read_mult"]
                          + r["uncached_in"] * r["p_in"] + r["base_out"]) * r["runs"]
            delta = r["warm"] - would_warm
            saved += delta
            print("  %-26s %s -> %s per task"
                  % (r["name"][:26], money(r["warm"]), money(would_warm)))
        print("")
        print("  %s per task, if those prefixes were pushed above the minimum."
              % money(saved))
        print("")

    print(line)
    print("  These are modelled figures from the token counts you supplied, not")
    print("  measurements. Check them against one real invoice before quoting")
    print("  them to anybody.")
    print(line)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    if len(sys.argv) != 2:
        print(__doc__.split("Usage:")[0].strip())
        print("Usage: python taskcost.py pipeline.txt")
        return 2
    try:
        with open(sys.argv[1]) as f:
            doc = parse(f.read())
        report(doc)
    except (ValueError, OSError) as e:
        print("error: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
