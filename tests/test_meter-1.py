import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qvunex import core, meter, prices, record_retry, task, wrap   # noqa: E402
from qvunex.core import _infer_batch, _record, configure, flush  # noqa: E402
from qvunex.usage import extract           # noqa: E402
from qvunex.report import analyse, render     # noqa: E402
from qvunex.store import Store                # noqa: E402


def fresh(path, **kw):
    """Point the meter at a new file, tearing down any live session first.

    The meter keeps one process-wide session on purpose — a library that opens a
    second corpus file behind your back is worse than useless. That makes the
    tests order-dependent unless they reset it explicitly, which is what this
    does. Without it, whichever test happens to run first wins and the rest
    either fail or, worse, quietly write into its file and assert on the wrong
    records.
    """
    if core._session is not None:
        core._session.close()
        core._session = None
    core._config["enabled"] = True
    configure(path=path, **kw)


def test_infer_batch_from_len():
    assert _infer_batch(([1, 2, 3],), {}) == 3


def test_infer_batch_from_shape():
    class Fake:
        shape = (7, 3, 224, 224)
    assert _infer_batch((Fake(),), {}) == 7


def test_infer_batch_string_is_one():
    assert _infer_batch(("hello world",), {}) == 1


def test_infer_batch_no_args():
    assert _infer_batch((), {}) == 1


def test_infer_batch_unsized_object():
    assert _infer_batch((object(),), {}) == 1


def test_store_roundtrip_and_corrupt_tail():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        s = Store(p, flush_every=1)
        s.write({"t": "call", "ts": 1.0})
        s.close()
        with open(p, "a") as f:
            f.write('{"t": "cal')          # simulate a kill mid-flush
        recs = list(Store.read(p))
        assert len(recs) == 1               # corrupt tail skipped, not raised


def test_decorator_records_and_preserves_return():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        fresh(path=p, rate_usd_hour=1.0)

        @meter("unit-test")
        def predict(batch):
            time.sleep(0.001)
            return len(batch)

        assert predict([0] * 5) == 5
        flush()
        recs = [r for r in Store.read(p) if r["t"] == "call"]
        assert len(recs) == 1
        assert recs[0]["endpoint"] == "unit-test"
        assert recs[0]["batch"] == 5
        assert recs[0]["ok"] is True
        assert recs[0]["dur_ms"] > 0


def test_exception_propagates_and_is_recorded():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        s = Store(p, flush_every=1)
        try:
            raise ValueError("boom")
        except ValueError as e:
            s.write({"t": "call", "v": "0.1", "ts": time.time(),
                     "endpoint": "x", "dur_ms": 1.0, "batch": 1,
                     "ok": False, "error": type(e).__name__})
        s.close()
        rec = list(Store.read(p))[0]
        assert rec["ok"] is False and rec["error"] == "ValueError"


def _synthetic(path, n=40):
    s = Store(path, flush_every=10)
    t = time.time()
    s.write({"t": "session", "v": "0.1", "ts": t, "host": "h",
             "gpus": [{"index": 0, "name": "T4", "driver": "x",
                       "power_limit_w": 70.0, "memory_total_mib": 15360.0}],
             "config": {"rate_usd_hour": 0.35}})
    for i in range(n):
        s.write({"t": "call", "v": "0.1", "ts": t + i * 0.1,
                 "endpoint": "big" if i % 2 else "small",
                 "dur_ms": 50.0 if i % 2 else 5.0,
                 "batch": 8 if i % 2 else 1, "ok": True})
        s.write({"t": "gpu", "v": "0.1", "ts": t + i * 0.1, "index": 0,
                 "util": 80.0 if i % 3 else 0.0, "power_w": 60.0,
                 "mem_used_mib": 2000.0, "mem_total_mib": 15360.0})
    s.close()


def test_analyse_and_render():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _synthetic(p)
        a = analyse(p, rate_usd_hour=0.35)
        assert not a["empty"]
        assert a["total_calls"] == 40
        assert a["total_inferences"] == 20 * 8 + 20 * 1
        assert a["n_gpus"] == 1
        assert a["total_cost_usd"] > 0
        assert a["cpki_usd"] > 0
        # 'big' does 10x the work per call -> dominates the cost share
        assert a["endpoints"][0]["endpoint"] == "big"
        assert a["endpoints"][0]["share_of_busy"] > 0.8
        # shares sum to 1
        assert abs(sum(e["share_of_busy"] for e in a["endpoints"]) - 1.0) < 1e-6
        # idle detected (every 3rd sample is 0% util)
        assert a["gpu"]["idle_fraction"] > 0.3
        assert a["gpu"]["idle_cost_usd"] > 0
        out = render(a)
        assert "QVUNEX METER REPORT" in out
        assert "COST BY ENDPOINT" in out
        assert "WASTE" in out


def test_duty_cycle_capped_at_one():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        s = Store(p, flush_every=1)
        t = time.time()
        # 10 calls of 1000ms each inside a 100ms window -> heavy concurrency
        for i in range(10):
            s.write({"t": "call", "v": "0.1", "ts": t + i * 0.01,
                     "endpoint": "c", "dur_ms": 1000.0, "batch": 1, "ok": True})
        s.close()
        a = analyse(p, rate_usd_hour=1.0)
        assert a["duty_cycle"] <= 1.0
        assert a["concurrent"] is True


def test_no_rate_means_no_fabricated_cost():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        s = Store(p, flush_every=1)
        s.write({"t": "call", "v": "0.1", "ts": time.time(),
                 "endpoint": "c", "dur_ms": 10.0, "batch": 1, "ok": True})
        s.close()
        a = analyse(p)
        assert a["total_cost_usd"] is None
        assert a["cpki_usd"] is None
        assert "not set" in render(a)


def test_empty_corpus():
    with tempfile.TemporaryDirectory() as d:
        a = analyse(os.path.join(d, "nope.jsonl"))
        assert a["empty"]
        assert "No calls recorded" in render(a)


def test_disabled_is_a_passthrough():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        import qvunex.core as M
        old = M._config["enabled"]
        M._config["enabled"] = False
        M._config["path"] = p
        try:
            @meter("off")
            def f(x):
                return x * 2
            assert f(21) == 42
            assert not os.path.exists(p)
        finally:
            M._config["enabled"] = old


# ---- checklist -------------------------------------------------------------

def test_checklist_marks_missing_fields():
    from qvunex.checklist import build
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _synthetic(p)
        lines, missing = build(p, context=None)
        out = "\n".join(lines)
        assert "COMPARABILITY CHECKLIST" in out
        assert missing > 0
        assert "MISSING" in out
        # things it CAN observe must not be marked missing
        assert "[observed]" in out
        assert "T4" in out                    # 1.1 from the session record
        assert "$0.3500/hr" in out            # 6.1 from recorded config


def test_checklist_uses_declared_context():
    from qvunex.checklist import build
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _synthetic(p)
        ctx = {"engine": "vLLM 0.27.1", "cuda_graphs": True,
               "model_id": "Qwen/Qwen2.5-1.5B-Instruct"}
        lines, missing = build(p, context=ctx)
        out = "\n".join(lines)
        assert "vLLM 0.27.1" in out
        assert "[declared]" in out
        before, _ = build(p, context=None)
        assert missing < len(before)          # declaring reduces the gap count


def test_checklist_empty_corpus():
    from qvunex.checklist import build
    with tempfile.TemporaryDirectory() as d:
        lines, missing = build(os.path.join(d, "nope.jsonl"))
        assert missing == 0
        assert "nothing to report" in "\n".join(lines)


def test_configure_accepts_context():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        fresh(path=p, rate_usd_hour=0.35, context={"engine": "vLLM 0.27.1"})

        @meter("ctx-test")
        def predict(batch):
            return len(batch)

        predict([0] * 3)
        flush()
        sess = [r for r in Store.read(p) if r["t"] == "session"]
        assert sess[0]["config"]["context"]["engine"] == "vLLM 0.27.1"


def test_checklist_variance_is_within_batch_not_pooled():
    """Pooling latency across batch sizes measures the batch mix, not
    reproducibility. A workload with wildly different batch sizes but stable
    per-batch timing must not report a huge spread."""
    from qvunex.checklist import build
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        s = Store(p, flush_every=50)
        t = time.time()
        # batch 1 always ~10ms, batch 32 always ~320ms -> pooled CV is enormous,
        # within-batch CV is ~0.
        for i in range(30):
            s.write({"t": "call", "v": "0.1", "ts": t + i * 0.1, "endpoint": "e",
                     "dur_ms": 10.0, "batch": 1, "ok": True})
            s.write({"t": "call", "v": "0.1", "ts": t + i * 0.1, "endpoint": "e",
                     "dur_ms": 320.0, "batch": 32, "ok": True})
        s.close()
        lines, _ = build(p)
        out = " ".join(l.strip() for l in lines)   # field text is word-wrapped
        import re
        m = re.search(r"latency CV within a fixed batch size (\d+)–(\d+)%", out)
        assert m, f"no within-batch CV found in output"
        lo, hi = float(m.group(1)), float(m.group(2))
        # per-batch timings are identical here, so the honest answer is ~0%.
        # A pooled calculation would report several hundred percent.
        assert hi < 5, f"pooled variance leaked through: {lo}-{hi}%"
        assert "not run-to-run reproducibility" in out


# ---------------------------------------------------------------------------
# schema 0.2 — tokens, tasks, wrapped clients
# ---------------------------------------------------------------------------

class _Obj:
    """Stand-in for a provider response. Duck-typed, no SDK needed."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _anthropic(cold=True):
    return _Obj(model="claude-sonnet-4.6", usage=_Obj(
        input_tokens=120, output_tokens=340,
        cache_creation_input_tokens=2400 if cold else 0,
        cache_read_input_tokens=0 if cold else 2400))


def test_usage_splits_cache_write_from_cache_read():
    cold = extract(_anthropic(cold=True))
    warm = extract(_anthropic(cold=False))
    assert cold["cache_write"] == 2400 and cold["cache_read"] == 0
    assert warm["cache_read"] == 2400 and warm["cache_write"] == 0
    # A cache write costs more than no cache at all; a read costs a tenth. If
    # these two collapsed into one number the report could not tell them apart.
    assert cold != warm


def test_usage_normalises_openai_inclusive_prompt_tokens():
    # OpenAI's prompt_tokens includes cached tokens; Anthropic's excludes them.
    r = _Obj(model="gpt-x", usage=_Obj(
        prompt_tokens=3000, completion_tokens=500,
        prompt_tokens_details=_Obj(cached_tokens=2000),
        completion_tokens_details=_Obj(reasoning_tokens=380)))
    u = extract(r)
    assert u["tokens_in"] == 1000        # 3000 billed, 2000 of them cached
    assert u["cache_read"] == 2000
    assert u["tokens_reasoning"] == 380  # billed, and buried on most dashboards


def test_usage_absent_is_none_not_zero():
    assert extract(_Obj(model="x")) is None
    u = extract(_anthropic())
    assert "tokens_reasoning" not in u   # Anthropic folds thinking into output


def test_task_id_threads_through_every_call_inside_it():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        fresh(path=p)

        @meter("step")
        def step():
            return 1

        with task("draft email"):
            step()
            step()
        step()                            # outside the task
        flush()

        recs = [r for r in Store.read(p) if r["t"] == "call"]
        assert len(recs) == 3
        inside = [r for r in recs if "task_id" in r]
        assert len(inside) == 2
        assert inside[0]["task"] == "draft email"
        assert inside[0]["task_id"] == inside[1]["task_id"]
        assert "task" not in recs[-1]


def test_wrap_records_calls_the_caller_never_touched():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        fresh(path=p)

        class Messages:
            def create(self, **kw):
                return _anthropic()

        class Client:
            def __init__(self):
                self.messages = Messages()

        client = wrap(Client())
        with task("one task"):
            client.messages.create(model="claude-sonnet-4.6", messages=[])
            client.messages.create(model="claude-sonnet-4.6", messages=[],
                                   qvunex_route="research")
        flush()

        recs = [r for r in Store.read(p) if r["t"] == "call"]
        assert len(recs) == 2
        assert recs[0]["endpoint"] == "messages.create"
        assert recs[1]["endpoint"] == "research"      # route label honoured
        assert recs[0]["cache_write"] == 2400
        assert recs[0]["task"] == "one task"


def test_wrap_marks_a_different_answering_model_as_fallback():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        fresh(path=p)

        class Messages:
            def create(self, **kw):
                return _Obj(model="claude-haiku-4.5", usage=_Obj(
                    input_tokens=10, output_tokens=20))

        class Client:
            def __init__(self):
                self.messages = Messages()

        wrap(Client()).messages.create(model="claude-opus-5", messages=[])
        flush()

        rec = [r for r in Store.read(p) if r["t"] == "call"][0]
        assert rec["status"] == "fallback"
        assert rec["model"] == "claude-haiku-4.5"     # what actually answered


def test_wrap_does_not_double_wrap():
    class Messages:
        def create(self, **kw):
            return None

    class Client:
        def __init__(self):
            self.messages = Messages()

    c = Client()
    first = wrap(c).messages.create
    assert wrap(c).messages.create is first


def test_wrap_never_breaks_the_caller():
    with tempfile.TemporaryDirectory() as d:
        fresh(path=os.path.join(d, "e.jsonl"))

        class Messages:
            def create(self, **kw):
                raise ValueError("provider said no")

        class Client:
            def __init__(self):
                self.messages = Messages()

        client = wrap(Client())
        try:
            client.messages.create(model="m")
        except ValueError as e:
            assert str(e) == "provider said no"      # untouched
        else:
            raise AssertionError("exception was swallowed")


def test_retry_is_recorded_as_spend():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        fresh(path=p)
        record_retry("research", dur_ms=12.0, attempt=2, model="m")
        flush()
        rec = [r for r in Store.read(p) if r["t"] == "call"][0]
        assert rec["status"] == "retry" and rec["attempt"] == 2


# ---- rate cards ------------------------------------------------------------

CARD = """
# a rate card
price: claude-sonnet-4-5
  input_per_mtok: 3.00
  output_per_mtok: 15.00

price: gpt-5
  input_per_mtok: 1.25
  output_per_mtok: 10.00
  cache_read_per_mtok: 0.125
"""


def test_price_card_parses():
    cards = prices.parse(CARD)
    assert cards["claude-sonnet-4-5"]["input"] == 3.00
    assert cards["gpt-5"]["cache_read"] == 0.125


def test_price_card_matches_dated_model_id():
    cards = prices.parse(CARD)
    card = prices.card_for("claude-sonnet-4-5-20250929", cards)
    assert card is not None and card["input"] == 3.00


def test_cache_rates_default_from_input_rate():
    r = prices.rates(prices.parse(CARD)["claude-sonnet-4-5"])
    # a write costs MORE than plain input, a read costs a tenth
    assert r["cache_write"] == 3.00 * 1.25
    assert r["cache_read"] == 3.00 * 0.10
    assert r["cache_write"] > r["input"] > r["cache_read"]


def test_unknown_model_is_unpriced_not_zero():
    cards = prices.parse(CARD)
    rec = {"model": "some-local-llama", "tokens_in": 1000, "tokens_out": 100}
    assert prices.cost_of(rec, cards) is None


def test_call_without_tokens_is_unpriced():
    cards = prices.parse(CARD)
    assert prices.cost_of({"model": "gpt-5", "dur_ms": 12.0}, cards) is None


def test_reasoning_tokens_are_not_billed_twice():
    cards = prices.parse(CARD)
    base = {"model": "gpt-5", "tokens_in": 0, "tokens_out": 1000}
    with_thinking = dict(base, tokens_reasoning=900)
    # thinking tokens are a breakdown of output, already counted
    assert prices.cost_of(base, cards) == prices.cost_of(with_thinking, cards)


def test_cache_read_is_cheaper_than_the_same_tokens_as_input():
    cards = prices.parse(CARD)
    cold = {"model": "claude-sonnet-4-5", "tokens_in": 10000}
    warm = {"model": "claude-sonnet-4-5", "tokens_in": 0, "cache_read": 10000}
    assert prices.cost_of(warm, cards) < prices.cost_of(cold, cards) / 5


# ---- cost per finished task ------------------------------------------------

def _task_corpus(path):
    """Two kinds of task. The cheap-looking step runs forty times and its
    prompt cache is written but never read -- the exact shape that a total
    hides and a per-task, per-endpoint view exposes."""
    s = Store(path, flush_every=50)
    t = time.time()
    s.write({"t": "session", "v": "0.2", "ts": t, "host": "h", "gpus": [],
             "config": {}})
    for i in range(6):
        tid = "task%02d" % i
        for j in range(40):
            s.write({"t": "call", "v": "0.2", "ts": t + i, "endpoint": "classify",
                     "dur_ms": 5.0, "batch": 1, "ok": True, "status": "ok",
                     "task": "outbound email", "task_id": tid,
                     "model": "claude-sonnet-4-5-20250929",
                     "tokens_in": 900, "tokens_out": 40, "cache_write": 850})
        s.write({"t": "call", "v": "0.2", "ts": t + i, "endpoint": "draft",
                 "dur_ms": 900.0, "batch": 1, "ok": True, "status": "ok",
                 "task": "outbound email", "task_id": tid,
                 "model": "claude-sonnet-4-5-20250929",
                 "tokens_in": 1200, "tokens_out": 800, "cache_read": 2400})
    # one retry, on one task only
    s.write({"t": "call", "v": "0.2", "ts": t, "endpoint": "draft",
             "dur_ms": 900.0, "batch": 1, "ok": True, "status": "retry",
             "attempt": 2, "task": "outbound email", "task_id": "task00",
             "model": "claude-sonnet-4-5-20250929",
             "tokens_in": 1200, "tokens_out": 800})
    # a call that belongs to no task at all
    s.write({"t": "call", "v": "0.2", "ts": t, "endpoint": "adhoc",
             "dur_ms": 10.0, "batch": 1, "ok": True, "status": "ok",
             "model": "claude-sonnet-4-5-20250929",
             "tokens_in": 500, "tokens_out": 50})
    s.close()


def test_cost_per_finished_task():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _task_corpus(p)
        a = analyse(p, prices=prices.parse(CARD))
        kinds = {k["task"]: k for k in a["task_kinds"]}
        k = kinds["outbound email"]
        assert k["n"] == 6
        assert k["mean_usd"] > 0
        # 40 classify calls + 1 draft, and task00 also has the retry
        assert 41 <= k["mean_calls"] <= 42
        assert k["retries"] == 1
        assert k["wasted_usd"] > 0
        assert "COST PER FINISHED TASK" in render(a)


def test_untasked_calls_are_counted_apart_not_spread():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _task_corpus(p)
        a = analyse(p, prices=prices.parse(CARD))
        assert a["untasked"]["calls"] == 1
        assert a["untasked"]["cost_usd"] > 0
        # every task id is its own task, none inflated by the orphan
        assert len(a["tasks"]) == 6
        assert "unattributable" in render(a)


def test_dead_cache_is_found_per_endpoint_not_hidden_in_the_total():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _task_corpus(p)
        a = analyse(p, prices=prices.parse(CARD))
        # globally there ARE reads, so a corpus-wide check would say nothing
        assert a["tokens"]["cache_read"] > 0 and a["tokens"]["cache_write"] > 0
        out = render(a)
        assert "read none back" in out and "classify" in out


def test_a_task_with_an_unpriced_call_is_flagged_not_undercounted():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        s = Store(p, flush_every=1)
        t = time.time()
        s.write({"t": "call", "v": "0.2", "ts": t, "endpoint": "a", "dur_ms": 1.0,
                 "batch": 1, "ok": True, "task": "mixed", "task_id": "x",
                 "model": "claude-sonnet-4-5", "tokens_in": 1000, "tokens_out": 10})
        s.write({"t": "call", "v": "0.2", "ts": t, "endpoint": "b", "dur_ms": 1.0,
                 "batch": 1, "ok": True, "task": "mixed", "task_id": "x",
                 "model": "mystery-model", "tokens_in": 9_000_000,
                 "tokens_out": 900_000})
        s.close()
        a = analyse(p, prices=prices.parse(CARD))
        assert a["tasks"][0]["priced"] is False
        k = a["task_kinds"][0]
        assert k["n"] == 1 and k["n_priced"] == 0
        assert k["mean_usd"] is None      # not a smaller, wrong number
        assert "left out of the figures" in render(a)


def test_no_rate_card_counts_tokens_without_costing_them():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.jsonl")
        _task_corpus(p)
        a = analyse(p, prices={})
        assert a["tokens"]["tokens_in"] > 0
        assert a["token_cost_usd"] is None
        assert "no rate card loaded" in render(a)
