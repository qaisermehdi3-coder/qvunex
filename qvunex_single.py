#!/usr/bin/env python3
"""
qvunex_single.py — one file, no install, no dependencies.

Measures what each LLM call actually costs and what a finished task costs.

    WHY ONE FILE
    ------------
    Several people told me the same thing: they will not pip install a package
    from a stranger onto a machine that holds production credentials. Fair. So
    this is one file you can read in ten minutes and paste into your own tree.
    Standard library only. No network code anywhere — grep it. Nothing leaves
    the machine; it appends JSON lines to a file on your own disk.

    USE
    ---
        from qvunex_single import wrap, task, step, report

        client = wrap(anthropic.Anthropic())      # or OpenAI, or Google

        with task("outbound email"):
            with step("research"):
                ...
            with step("draft"):
                ...

        report()          # prints cost per finished task

    Try it with no API key and no network:

        python3 qvunex_single.py --demo

    RATE CARD
    ---------
    Costs need prices. Put a plain text file at ~/.qvunex/prices.txt:

        # model              input   output  cache_write  cache_read
        claude-sonnet-4-5    3.00    15.00   3.75         0.30
        gpt-5                1.25    10.00   1.5625       0.125
        gemini-3.6-flash     0.30    2.50    0.375        0.03

    Dollars per million tokens. A model with no entry is reported as unpriced
    and left out rather than estimated.

    Apache-2.0. https://github.com/qaisermehdi3-coder/qvunex
"""

from __future__ import annotations

import contextvars
import json
import os
import statistics
import sys
import time
import uuid
from contextlib import contextmanager

SCHEMA_VERSION = "0.3"
DEFAULT_PATH = os.path.expanduser("~/.qvunex/events.jsonl")
DEFAULT_PRICES = os.path.expanduser("~/.qvunex/prices.txt")

# How a provider bills reasoning / thinking tokens. Recorded per call, never
# assumed globally, because providers disagree and none of them raise an error
# if you guess wrong.
INCLUDED = "included"   # already inside the output count; do not add again
EXTRA = "extra"         # billed on top of the output count; must be added

CACHE_WRITE_MULT_5M = 1.25   # a 5-minute cache write costs MORE than not caching
CACHE_WRITE_MULT_1H = 2.00
CACHE_READ_MULT = 0.10


# ---------------------------------------------------------------------------
# task / step attribution
# ---------------------------------------------------------------------------
# A ContextVar holding a tuple of (label, id) pairs, outermost first.
#
# WHAT INHERITS, tested rather than assumed:
#
#   asyncio tasks   YES. Each task copies the context at creation, so concurrent
#                   tasks keep separate labels and cannot leak into each other.
#   threads         NO. A new thread starts with a fresh, empty context, so it
#                   sees no task and no step. Calls made in a worker thread are
#                   recorded with task=None and land in the "outside any task"
#                   line - honest, but not attributed.
#   processes       NO.
#
# The thread case matters because many frameworks run tool calls and sub-agent
# calls on a ThreadPoolExecutor. If you own the thread, wrap the callable with
# contextvars.copy_context().run(fn) and the labels carry over. If a framework
# owns the pool, you cannot, so pass the id yourself:
#
#     tid = ...                      # captured in the parent
#     with task("my task", task_id=tid):
#         ...
#
# Exceptions are safe in all cases: every label is popped in a finally block, so
# a throw inside a step restores the stack rather than leaving a label stuck.

_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar("qvunex_stack", default=())


@contextmanager
def task(name, task_id=None):
    """Name a unit of work. Cost per finished task is measured against this."""
    tid = task_id or uuid.uuid4().hex[:12]
    token = _stack.set(_stack.get() + ((name, tid),))
    try:
        yield tid
    finally:
        _stack.reset(token)


@contextmanager
def step(name):
    """Name a step inside a task, so calls can be attributed to it.

    Without this, calls made inside a task are recorded as UNATTRIBUTED rather
    than silently credited to the task as a whole. A total that looks complete
    and isn't is worse than a gap you can see.
    """
    token = _stack.set(_stack.get() + ((name, None),))
    try:
        yield
    finally:
        _stack.reset(token)


def _current():
    s = _stack.get()
    if not s:
        return None, None, None
    task_name, task_id = s[0]
    owner = s[-1][0] if len(s) > 1 else None
    return task_name, task_id, owner


# ---------------------------------------------------------------------------
# reading usage off a provider response
# ---------------------------------------------------------------------------
# Duck-typed on purpose. No SDK imports, so this file works whether or not you
# have any particular client library installed, and does not break when one of
# them changes its package layout.

def _get(obj, *names):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def extract_usage(response):
    """Return a dict of token counts, or None if this isn't a usage-bearing response."""
    u = _get(response, "usage", "usage_metadata")
    if u is None:
        return None

    out = {}

    # ---- Anthropic shape -------------------------------------------------
    if _get(u, "input_tokens") is not None:
        out["tokens_in"] = _get(u, "input_tokens") or 0
        out["tokens_out"] = _get(u, "output_tokens") or 0
        out["cache_write"] = _get(u, "cache_creation_input_tokens") or 0
        out["cache_read"] = _get(u, "cache_read_input_tokens") or 0
        # Anthropic folds thinking into output_tokens and never reports it apart.
        out["tokens_reasoning"] = 0
        out["reasoning_billed"] = INCLUDED
        # input_tokens EXCLUDES cached tokens on Anthropic.
        total = out["tokens_in"] + out["cache_write"] + out["cache_read"] + out["tokens_out"]
        out["tokens_total_reported"] = total

    # ---- Google shape ----------------------------------------------------
    elif _get(u, "prompt_token_count") is not None:
        out["tokens_in"] = _get(u, "prompt_token_count") or 0
        out["tokens_out"] = _get(u, "candidates_token_count") or 0
        out["cache_read"] = _get(u, "cached_content_token_count") or 0
        out["cache_write"] = 0
        thoughts = _get(u, "thoughts_token_count") or 0
        out["tokens_reasoning"] = thoughts
        # Measured on live gemini-3.6-flash calls, 22 September 2026, against
        # Google's own reported total:
        #     17 +   3 + 135 =   155
        #     19 +  81 + 776 =   876
        #     15 + 397 + 923 = 1,335
        # total = prompt + candidates + thoughts, exactly, every time.
        # Worth seeing the first one: the answer was 3 tokens and the thinking
        # was 135. Anything charting candidates_token_count as the output cost
        # reports 3 where 138 was billed.
        # So on Google, thinking is billed ON TOP of the output count.
        out["reasoning_billed"] = EXTRA if thoughts else INCLUDED
        out["tokens_total_reported"] = _get(u, "total_token_count")

    # ---- OpenAI shape ----------------------------------------------------
    elif _get(u, "prompt_tokens") is not None:
        out["tokens_in"] = _get(u, "prompt_tokens") or 0
        out["tokens_out"] = _get(u, "completion_tokens") or 0
        details = _get(u, "completion_tokens_details") or {}
        out["tokens_reasoning"] = _get(details, "reasoning_tokens") or 0
        # reasoning_tokens is a SUBSET of completion_tokens. Adding it again
        # inflates the bill.
        out["reasoning_billed"] = INCLUDED
        pd = _get(u, "prompt_tokens_details") or {}
        out["cache_read"] = _get(pd, "cached_tokens") or 0
        out["cache_write"] = 0
        # prompt_tokens is INCLUSIVE of cached tokens on OpenAI, so subtract
        # them back out to avoid counting the same tokens twice.
        out["tokens_in"] = max(0, out["tokens_in"] - out["cache_read"])
        out["tokens_total_reported"] = _get(u, "total_tokens")

    else:
        return None

    _reconcile(out)
    return out


def _reconcile(out):
    """Compare our parts against the provider's own total.

    That total is the one number we did not derive. Where the parts don't add
    up, record the gap. Never correct it: we do not know which side is wrong,
    and quietly adjusting a number to make a check pass is how a meter starts
    lying.
    """
    total = out.get("tokens_total_reported")
    if total is None:
        return
    accounted = (out.get("tokens_in", 0) + out.get("cache_read", 0)
                 + out.get("cache_write", 0) + out.get("tokens_out", 0))
    if out.get("reasoning_billed") == EXTRA:
        accounted += out.get("tokens_reasoning", 0)
    diff = total - accounted
    if diff:
        out["tokens_unaccounted"] = diff


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.path = os.environ.get("QVUNEX_PATH", DEFAULT_PATH)
        self.disabled = os.environ.get("QVUNEX_DISABLED") == "1"
        self._started = False

    def write(self, rec):
        if self.disabled:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            if not self._started:
                self._started = True
                with open(self.path, "a") as f:
                    f.write(json.dumps({
                        "type": "session",
                        "schema": SCHEMA_VERSION,
                        "started": time.time(),
                    }) + "\n")
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            # A measurement tool that can take down production is a tool nobody
            # installs twice. Never raise out of here.
            pass


_rec = _Recorder()


def configure(path=None, disabled=None):
    if path is not None:
        _rec.path = os.path.expanduser(path)
        _rec._started = False
    if disabled is not None:
        _rec.disabled = bool(disabled)


def _record(model, usage, seconds, status, attempt):
    task_name, task_id, owner = _current()
    rec = {
        "type": "call",
        "schema": SCHEMA_VERSION,
        "t": time.time(),
        "model": model,
        "task": task_name,
        "task_id": task_id,
        # If nothing named the step, the owner is unknown. It is recorded as
        # unattributed and shown on its own line, never folded into the parent.
        "owner": owner,
        "seconds": round(seconds, 6),
        "status": status,
        "attempt": attempt,
    }
    if usage:
        rec.update(usage)
    _rec.write(rec)


# ---------------------------------------------------------------------------
# wrapping a client
# ---------------------------------------------------------------------------
# Wrapping the client matters more than decorating your own functions.
# Decorating only sees the calls you wrote; wrapping also catches the ones your
# framework makes on your behalf - which is where the spend hides when one
# request fans out into six sub-agent calls.

CALL_METHODS = (
    "create", "acreate",
    "generate_content", "generate_content_async",
    "generate", "complete", "invoke",
)


class _Proxy:
    def __init__(self, inner, path=""):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_inner"), name)
        path = object.__getattribute__(self, "_path")
        newpath = f"{path}.{name}" if path else name
        # Only terminal call methods are timed. Names like `messages`,
        # `chat` and `completions` are namespaces on their way to a call,
        # even when they are themselves callable - wrapping those would
        # record the lookup instead of the request.
        if callable(attr) and name in CALL_METHODS:
            return _timed(attr, newpath)
        if hasattr(attr, "__dict__") or hasattr(attr, "create"):
            return _Proxy(attr, newpath)
        return attr

    def __call__(self, *a, **k):
        return object.__getattribute__(self, "_inner")(*a, **k)


def _timed(fn, path):
    def inner(*a, **k):
        model = k.get("model") or "unknown"
        t0 = time.perf_counter()
        attempt = int(k.pop("_qvunex_attempt", 1))
        try:
            resp = fn(*a, **k)
        except Exception:
            _record(model, None, time.perf_counter() - t0, "error", attempt)
            raise          # your exception propagates untouched
        _record(model, extract_usage(resp), time.perf_counter() - t0, "ok", attempt)
        return resp
    return inner


def wrap(client):
    """Wrap a provider client so every call through it is recorded."""
    return _Proxy(client)


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def load_prices(path=None):
    path = path or os.environ.get("QVUNEX_PRICES", DEFAULT_PRICES)
    cards = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                name = parts[0]
                nums = [float(x) for x in parts[1:]]
                card = {"input": nums[0], "output": nums[1]}
                card["cache_write"] = nums[2] if len(nums) > 2 else nums[0] * CACHE_WRITE_MULT_5M
                card["cache_read"] = nums[3] if len(nums) > 3 else nums[0] * CACHE_READ_MULT
                cards[name] = card
    except FileNotFoundError:
        pass
    return cards


def cost_of(rec, cards):
    """Dollars for one call, or None if the model has no rate card entry."""
    card = cards.get(rec.get("model"))
    if not card:
        return None
    billable = [("tokens_in", "input"), ("tokens_out", "output"),
                ("cache_write", "cache_write"), ("cache_read", "cache_read")]
    if rec.get("reasoning_billed") == EXTRA:
        billable.append(("tokens_reasoning", "output"))
    total = 0.0
    for field, price_key in billable:
        total += (rec.get(field, 0) or 0) * card.get(price_key, 0.0) / 1_000_000
    return total


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def report(path=None, prices=None, out=sys.stdout):
    path = path or _rec.path
    cards = load_prices(prices)

    calls = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == "call":
                    calls.append(r)
    except FileNotFoundError:
        print(f"no corpus at {path}", file=out)
        return

    if not calls:
        print(f"no calls recorded in {path}", file=out)
        return

    w = lambda s="": print(s, file=out)
    w("=" * 68)
    w("  QVUNEX REPORT")
    w("=" * 68)
    w(f"  corpus        {path}")
    w(f"  calls         {len(calls):,}")

    unpriced = sorted({c.get("model") for c in calls if c.get("model") not in cards})
    unaccounted = sum(c.get("tokens_unaccounted", 0) or 0 for c in calls)

    # ---- cost per finished task -----------------------------------------
    tasks = {}
    untasked = []
    for c in calls:
        tid = c.get("task_id")
        if tid:
            tasks.setdefault((c.get("task"), tid), []).append(c)
        else:
            untasked.append(c)

    if tasks:
        by_name = {}
        for (name, _tid), cs in tasks.items():
            cost = sum(cost_of(c, cards) or 0 for c in cs)
            by_name.setdefault(name, []).append((cost, len(cs)))
        w()
        w("-" * 68)
        w("  COST PER FINISHED TASK")
        w("-" * 68)
        w("  %-22s %4s %10s %10s %10s %7s" % ("task", "n", "mean", "p50", "p95", "calls"))
        for name, rows in sorted(by_name.items(), key=lambda kv: -sum(r[0] for r in kv[1])):
            costs = [r[0] for r in rows]
            w("  %-22s %4d %10s %10s %10s %7.1f" % (
                name[:22], len(costs),
                f"${statistics.mean(costs):.4f}",
                f"${_pct(costs, 0.50):.4f}",
                f"${_pct(costs, 0.95):.4f}",
                statistics.mean([r[1] for r in rows])))
        w()
        w("  The mean and the p95 are different questions. An average hides the")
        w("  tasks that actually hurt.")

    if untasked:
        cost = sum(cost_of(c, cards) or 0 for c in untasked)
        w()
        w("  %d call(s) outside any task, $%.4f — counted apart, never spread" %
          (len(untasked), cost))

    # ---- attribution ------------------------------------------------------
    inside = [c for c in calls if c.get("task_id")]
    unattributed = [c for c in inside if not c.get("owner")]
    if inside:
        w()
        w("-" * 68)
        w("  ATTRIBUTION")
        w("-" * 68)
        owners = {}
        for c in inside:
            owners.setdefault(c.get("owner") or "UNATTRIBUTED", []).append(c)
        for owner, cs in sorted(owners.items(), key=lambda kv: -sum(cost_of(c, cards) or 0 for c in kv[1])):
            cost = sum(cost_of(c, cards) or 0 for c in cs)
            w("  %-30s %6d calls   $%.4f" % (owner[:30], len(cs), cost))
        if unattributed:
            w()
            w("  UNATTRIBUTED means the call happened inside a task but nothing")
            w("  named the step. It is counted, never folded into the parent.")
            w("  If your framework does not say which sub-agent is calling, no")
            w("  amount of wrapping can invent it.")

    # ---- cache ------------------------------------------------------------
    cw = sum(c.get("cache_write", 0) or 0 for c in calls)
    cr = sum(c.get("cache_read", 0) or 0 for c in calls)
    if cw or cr:
        w()
        w("-" * 68)
        w("  CACHE")
        w("-" * 68)
        w(f"  written {cw:,} tokens    read back {cr:,} tokens")
        if cw and not cr:
            w()
            w("  Written and never read. A 5-minute write costs 1.25x input and a")
            w("  1-hour write costs 2x, while a read costs 0.1x — so this is a")
            w("  pure surcharge over not caching at all.")

    # ---- retries and fallbacks -------------------------------------------
    retries = [c for c in calls if (c.get("attempt") or 1) > 1]
    if retries:
        cost = sum(cost_of(c, cards) or 0 for c in retries)
        w()
        w("  retries               %d call(s), $%.4f — the same work paid for twice"
          % (len(retries), cost))

    # ---- honesty ----------------------------------------------------------
    if unaccounted:
        w()
        w("-" * 68)
        w(f"  UNACCOUNTED  {unaccounted:,} tokens")
        w("-" * 68)
        w("  The provider's own reported total does not match the parts we")
        w("  recorded. Every cost above is short by at least this much. The gap")
        w("  is recorded, never corrected — we do not know which side is wrong.")

    if unpriced:
        w()
        w("  unpriced models (no rate card entry, left out rather than estimated):")
        for m in unpriced:
            w(f"    {m}")

    w()


# ---------------------------------------------------------------------------
# demo — no key, no network
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, usage):
        self.usage = usage


class _FakeAnthropic:
    class messages:
        @staticmethod
        def create(model=None, **k):
            time.sleep(0.001)
            return _FakeResp({
                "input_tokens": 900, "output_tokens": 120,
                "cache_creation_input_tokens": 3400,
                "cache_read_input_tokens": 0,
            })


class _FakeGoogle:
    class models:
        @staticmethod
        def generate_content(model=None, **k):
            time.sleep(0.001)
            # thoughts are billed ON TOP of candidates on Google
            return _FakeResp(None) if False else type("R", (), {
                "usage_metadata": {
                    "prompt_token_count": 17,
                    "candidates_token_count": 144,
                    "thoughts_token_count": 589,
                    "total_token_count": 750,
                }})()


def _demo():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "demo.jsonl")
    configure(path=path)

    prices = os.path.join(os.path.dirname(path), "prices.txt")
    with open(prices, "w") as f:
        f.write("claude-demo   3.00  15.00  3.75  0.30\n")
        f.write("gemini-demo   0.30   2.50  0.375 0.03\n")

    a = wrap(_FakeAnthropic())
    g = wrap(_FakeGoogle())

    for i in range(6):
        with task("outbound email"):
            with step("research"):
                g.models.generate_content(model="gemini-demo")
            with step("draft"):
                a.messages.create(model="claude-demo")
            # deliberately unnamed: this is what an un-instrumented sub-agent
            # call looks like in the report
            a.messages.create(model="claude-demo")

    with task("support triage"):
        a.messages.create(model="claude-demo", _qvunex_attempt=2)

    report(path=path, prices=prices)
    print(f"corpus written to {path}\n")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    elif len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        report(path=sys.argv[1])
    else:
        print(__doc__)
