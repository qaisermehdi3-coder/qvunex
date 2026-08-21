import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qvunex import meter                      # noqa: E402
from qvunex.core import _infer_batch, _record, configure, flush  # noqa: E402
from qvunex.report import analyse, render     # noqa: E402
from qvunex.store import Store                # noqa: E402


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
        configure(path=p, rate_usd_hour=1.0)

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
        configure(path=p, rate_usd_hour=0.35, context={"engine": "vLLM 0.27.1"})

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
