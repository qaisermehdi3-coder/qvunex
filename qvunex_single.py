#!/usr/bin/env python3
"""
qvunex_single.py — one file, no install, no dependencies.

Measures what each LLM call actually costs and what a finished task costs.

    WHY ONE FILE
    ------------
    Several people told me the same thing: they will not pip install a package
    from a stranger onto a machine that holds production credentials. Fair. So
    this is one file you can read in one sitting and paste into your own tree:
    about 900 lines, of which roughly 200 are the --selftest at the bottom.
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

    Check every behaviour described here on your own machine:

        python3 qvunex_single.py --selftest

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
#   threads         NO on Python 3.10 to 3.13, which is what has been tested. A
#                   new thread starts with a fresh, empty context, so it sees no
#                   task and no step. Calls made in a worker thread are recorded
#                   with task=None and land in the "outside any task" line -
#                   honest, but not attributed. Newer Pythons may differ; run
#                   --selftest and it will tell you what yours does.
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
    elif status == "ok":
        # The call succeeded and the provider returned no usage object. Its cost
        # is UNKNOWN, not zero. Left unmarked, this row is byte-identical to a
        # call that genuinely cost nothing - and a total that silently counts it
        # as free reads low, which is the direction nobody audits.
        rec["usage_missing"] = True
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
        # A task only counts as finished if its LAST call succeeded. A task that
        # crashed part way is not a finished task that happened to be cheap -
        # averaging it in at its partial cost drags the mean down silently.
        # Found on live Gemini data: one task hit a rate limit on its first
        # call, was averaged in at $0, and understated the mean by a third.
        by_name = {}
        failed = {}
        for (name, _tid), cs in tasks.items():
            cost = sum(cost_of(c, cards) or 0 for c in cs)
            last = max(cs, key=lambda c: c.get("t", 0))
            if last.get("status") == "ok":
                by_name.setdefault(name, []).append((cost, len(cs)))
            else:
                failed.setdefault(name, []).append((cost, len(cs)))
        if by_name:
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
        if failed:
            w()
            for name, rows in sorted(failed.items()):
                w("  %d '%s' task(s) did not finish, $%.4f spent before failing —"
                  % (len(rows), name[:22], sum(r[0] for r in rows)))
            w("  counted apart, never averaged into the finished ones")

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
    errors = [c for c in calls if c.get("status") == "error"]
    if errors:
        w()
        w("  errors                %d call(s) failed — included in the call counts"
          % len(errors))
        w("                        above, but they returned no usage, so they cost")
        w("                        nothing that can be measured here")

    retries = [c for c in calls if (c.get("attempt") or 1) > 1]
    if retries:
        cost = sum(cost_of(c, cards) or 0 for c in retries)
        w()
        w("  retries               %d call(s), $%.4f — the same work paid for twice"
          % (len(retries), cost))

    # ---- honesty ----------------------------------------------------------
    missing = [c for c in calls if c.get("usage_missing")]
    if missing:
        w()
        w("-" * 68)
        w(f"  USAGE MISSING  {len(missing)} call(s)")
        w("-" * 68)
        by_owner = {}
        for c in missing:
            by_owner.setdefault(c.get("owner") or ("UNATTRIBUTED" if c.get("task_id") else "outside any task"), 0)
            by_owner[c.get("owner") or ("UNATTRIBUTED" if c.get("task_id") else "outside any task")] += 1
        for owner, n in sorted(by_owner.items(), key=lambda kv: -kv[1]):
            w("  %-30s %6d calls   cost unknown" % (owner[:30], n))
        w()
        w("  These calls succeeded but the provider returned no usage object.")
        w("  They are shown above at $0 because nothing was reported, not because")
        w("  they were free. Every total above is a lower bound. A low number is")
        w("  the one nobody audits, so it is printed here rather than left silent.")

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


# ---------------------------------------------------------------------------
# selftest — every behaviour claimed in public, checked on your own machine
# ---------------------------------------------------------------------------
# These check the METER's own logic: attribution, arithmetic, what the report
# says. They use small stand-in provider objects shaped like the documented
# responses, so they cannot prove a live provider still behaves that way.
# For that, wrap a real client and read the report.

def _selftest(out=sys.stdout):
    import asyncio, ast, io, tempfile, threading
    w = lambda s="": print(s, file=out)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        w(("  PASS  " if ok else "  FAIL  ") + name + (f"  ({detail})" if detail else ""))

    def fresh():
        d = tempfile.mkdtemp()
        p = os.path.join(d, "events.jsonl")
        configure(path=p)
        return p

    def calls_in(p):
        with open(p) as f:
            return [r for r in (json.loads(l) for l in f) if r.get("type") == "call"]

    def text_of(p, prices):
        buf = io.StringIO()
        report(path=p, prices=prices, out=buf)
        return buf.getvalue()

    def fake(usage=None, raises=None, attr="usage"):
        class Ns:
            @staticmethod
            def create(model=None, **k):
                if raises:
                    raise raises
                return type("R", (), {attr: usage} if usage is not None else {})()
        return type("Client", (), {"messages": Ns})()

    def close(a, b):
        return abs(a - b) < 1e-12

    saved = (_rec.path, _rec._started)
    d = tempfile.mkdtemp()
    prices = os.path.join(d, "prices.txt")
    with open(prices, "w") as f:
        f.write("anth 3.00 15.00 3.75 0.30\n"
                "oai  1.00  4.00 1.25 0.10\n"
                "goog 1.00 10.00 1.25 0.10\n")
    cards = load_prices(prices)

    w("=" * 68)
    w("  QVUNEX SELFTEST")
    w("=" * 68)
    w(f"  python {sys.version.split()[0]}")
    w()

    try:
        # -- provider arithmetic ------------------------------------------
        w("  provider arithmetic")
        p = fresh()
        wrap(fake({"input_tokens": 100, "output_tokens": 50,
                   "cache_creation_input_tokens": 400,
                   "cache_read_input_tokens": 1000})).messages.create(model="anth")
        c = calls_in(p)[0]
        check("anthropic: input excludes cache, cache priced on its own",
              close(cost_of(c, cards), (100*3 + 50*15 + 400*3.75 + 1000*0.30) / 1e6))

        p = fresh()
        wrap(fake({"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                   "prompt_tokens_details": {"cached_tokens": 800},
                   "completion_tokens_details": {"reasoning_tokens": 150}})).messages.create(model="oai")
        c = calls_in(p)[0]
        check("openai: cached tokens not counted twice, reasoning not added again",
              close(cost_of(c, cards), (200*1 + 800*0.10 + 200*4) / 1e6))

        p = fresh()
        wrap(fake({"prompt_token_count": 10, "candidates_token_count": 5,
                   "thoughts_token_count": 20, "total_token_count": 35},
                  attr="usage_metadata")).messages.create(model="goog")
        c = calls_in(p)[0]
        check("google: thinking billed on top of output",
              close(cost_of(c, cards), (10*1 + 5*10 + 20*10) / 1e6))

        p = fresh()
        wrap(fake({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 175})
             ).messages.create(model="oai")
        c = calls_in(p)[0]
        check("provider total that does not match the parts is recorded as a gap",
              c.get("tokens_unaccounted") == 25, f"gap {c.get('tokens_unaccounted')}")

        # -- attribution ---------------------------------------------------
        w()
        w("  attribution")
        p = fresh()
        cl = wrap(fake({"input_tokens": 10, "output_tokens": 5}))
        with task("t"):
            with step("named"):
                cl.messages.create(model="anth")
            cl.messages.create(model="anth")
        cs = calls_in(p)
        check("a call no step names stays unowned, never folded into the parent",
              [x.get("owner") for x in cs] == ["named", None])
        check("the report prints it on an UNATTRIBUTED line",
              "UNATTRIBUTED" in text_of(p, prices))

        p = fresh()
        with task("t"):
            try:
                with step("boom"):
                    raise RuntimeError("x")
            except RuntimeError:
                pass
            # checked INSIDE the task: the task's own cleanup would otherwise
            # hide a step that failed to remove its label
            after_step = _current()
        check("an exception inside a step leaves no stale label behind",
              after_step[0] == "t" and after_step[2] is None and _current() == (None, None, None))

        p = fresh()
        async def worker(name, delay):
            with task(name):
                with step(name + "-step"):
                    await asyncio.sleep(delay)
                    cl.messages.create(model="anth")
        async def both():
            await asyncio.gather(worker("a", 0.02), worker("b", 0.01))
        # run in its own thread so this works even inside a notebook's event loop
        th = threading.Thread(target=lambda: asyncio.run(both()))
        th.start(); th.join()
        pairs = sorted((x.get("task"), x.get("owner")) for x in calls_in(p))
        check("concurrent asyncio tasks keep their labels separate",
              pairs == [("a", "a-step"), ("b", "b-step")])

        seen = {}
        with task("parent"):
            with step("parent-step"):
                t = threading.Thread(target=lambda: seen.__setitem__("plain", _current()[0]))
                t.start(); t.join()
                ctx = contextvars.copy_context()
                t = threading.Thread(target=lambda: seen.__setitem__("copied", ctx.run(lambda: _current()[0])))
                t.start(); t.join()
        inherits = seen.get("plain") == "parent"
        check("contextvars.copy_context() carries labels into a thread you own",
              seen.get("copied") == "parent")
        w("  INFO  a plain new thread %s the task label on this Python"
          % ("INHERITS" if inherits else "does NOT inherit"))

        # -- report honesty ------------------------------------------------
        w()
        w("  report honesty")
        p = fresh()
        ok = wrap(fake({"input_tokens": 1000, "output_tokens": 100}))
        bad = wrap(fake(raises=RuntimeError("rate limited")))
        with task("job"):
            ok.messages.create(model="anth")
        try:
            with task("job"):
                bad.messages.create(model="anth")
        except RuntimeError:
            pass
        txt = text_of(p, prices)
        one = (1000*3 + 100*15) / 1e6
        check("a crashed task is not averaged in as a cheap finished one",
              f"${one:.4f}" in txt and "did not finish" in txt)

        p = fresh()
        caught = False
        try:
            bad.messages.create(model="anth")
        except RuntimeError as e:
            caught = str(e) == "rate limited"
        c = calls_in(p)[0]
        check("your exception reaches you untouched, and is recorded as an error",
              caught and c.get("status") == "error" and not c.get("usage_missing"))

        p = fresh()
        with task("t"):
            with step("s"):
                wrap(fake(None)).messages.create(model="anth")
        c = calls_in(p)[0]
        txt = text_of(p, prices)
        check("a call with no usage object is marked unknown, not counted as free",
              c.get("usage_missing") is True and "USAGE MISSING" in txt and "lower bound" in txt)

        # -- the claim anyone can grep -------------------------------------
        w()
        w("  what this file does not do")
        net = {"socket", "urllib", "http", "requests", "httpx", "aiohttp",
               "ssl", "ftplib", "smtplib", "urllib3", "websocket", "websockets"}
        with open(os.path.abspath(__file__)) as f:
            tree = ast.parse(f.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        bad_imports = sorted(imported & net)
        check("no networking module is imported anywhere in this file",
              not bad_imports, ", ".join(bad_imports) if bad_imports else "")
    finally:
        _rec.path, _rec._started = saved

    passed = sum(results)
    w()
    w(f"  {passed} of {len(results)} passed")
    w()
    w("  These test the meter's own logic against stand-in responses shaped")
    w("  like each provider's documented usage object. They cannot tell you a")
    w("  live provider still reports that shape. To check that, wrap your real")
    w("  client, make one call, and read the report.")
    w()
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    elif "--demo" in sys.argv:
        _demo()
    elif len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        report(path=sys.argv[1])
    else:
        print(__doc__)
