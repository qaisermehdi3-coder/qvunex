"""A synthetic workload, so the report can be seen before wiring up a real one.

Deliberately shaped like a workload with real problems: one dominant endpoint,
a cheap one that runs constantly, ragged batch sizes, and idle gaps. Those are
the things the report is meant to surface, so the demo should contain them.
"""

import os
import random
import time

from . import meter
from .core import configure, flush, task, wrap, record_retry, _config
from .prices import parse as parse_prices
from .report import analyse, render

# Made-up rates, so the demo can print money without shipping a rate card that
# goes stale. Nothing here should be copied into a real one -- use
# tools/prices.example.txt and put in the published numbers yourself.
DEMO_PRICES = """
price: demo-large
  input_per_mtok: 3.00
  output_per_mtok: 15.00
price: demo-small
  input_per_mtok: 1.00
  output_per_mtok: 5.00
price: demo-premium
  input_per_mtok: 5.00
  output_per_mtok: 25.00
price: demo-thinker
  input_per_mtok: 1.25
  output_per_mtok: 10.00
  cache_read_per_mtok: 0.125
"""


def _start_clean(path):
    """Demos write a fresh file each run.

    The store appends, which is right for a meter -- a corpus is meant to
    accumulate. It is wrong for a demo: run it twice and every number doubles,
    which is exactly the kind of quietly wrong figure this project cannot
    afford to print.
    """
    try:
        os.remove(path)
    except OSError:
        pass
    return path


def run_demo(seconds=12.0, rate=0.35):
    configure(path=_start_clean("./qvunex_demo.jsonl"), rate_usd_hour=rate,
              sample_interval=0.5)

    @meter("image-classifier")
    def classify(batch):
        time.sleep(0.010 + 0.0015 * len(batch))
        return [0] * len(batch)

    @meter("text-embedder")
    def embed(batch):
        time.sleep(0.004 * len(batch))
        return [[0.0]] * len(batch)

    @meter("thumbnail-scorer")
    def score(batch):
        time.sleep(0.001)
        return [0.5] * len(batch)

    rng = random.Random(7)
    t_end = time.time() + seconds
    print(f"running synthetic workload for {seconds:.0f}s ...")

    while time.time() < t_end:
        # ragged batches -- the batch-efficiency signal
        classify([0] * rng.choice([32, 8, 4, 32, 2, 16]))
        embed([0] * rng.choice([16, 4, 8]))
        for _ in range(3):
            score([0])
        # a quiet gap -- the idle-burn signal
        if rng.random() < 0.25:
            time.sleep(rng.uniform(0.3, 0.9))

    flush()
    print()
    print(render(analyse(_config["path"], rate)))
    print(f"raw events: {_config['path']}")
    return 0


# ---------------------------------------------------------------------------
# the per-token path
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Response:
    def __init__(self, model, usage):
        self.model = model
        self.usage = usage


class _Messages:
    """Stands in for a provider client. No network, no key, no SDK.

    The token counts are invented, but the *shapes* are not: the cheap step
    writes a cache prefix under the model's minimum, so it is charged for the
    write and never gets a read; the expensive step has a wide spread; and one
    call in twelve comes back from a different model than the one asked for.
    """

    def __init__(self, rng):
        self.rng = rng

    def create(self, **kw):
        model = kw.get("model", "demo-large")
        time.sleep(0.002)
        if model == "demo-small":
            return _Response(model, _Usage(
                input_tokens=900, output_tokens=40,
                cache_creation_input_tokens=850,   # under the minimum: never read
                cache_read_input_tokens=0))
        # one call in twelve is answered by a dearer model than the one asked
        # for: a rate-limit fallback, real spend that ordinary logging files
        # under the model you thought you were using.
        answered = "demo-large" if self.rng.random() > 0.08 else "demo-premium"
        return _Response(answered, _Usage(
            input_tokens=1200,
            output_tokens=self.rng.choice([400, 500, 600, 700, 2600]),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2400))


class _Client:
    def __init__(self, rng):
        self.messages = _Messages(rng)


class _OpenAIDetails:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _Completions:
    """A second provider, reporting the same facts in a different shape.

    Two differences that matter and are easy to miss: prompt_tokens here is
    *inclusive* of cached tokens where the other provider's input_tokens
    excludes them, and reasoning tokens are reported as a breakdown of the
    output. Both are handled in usage.py, which is why the two providers can
    appear in one report without the numbers meaning different things.
    """

    def __init__(self, rng):
        self.rng = rng

    def create(self, **kw):
        time.sleep(0.002)
        thinking = self.rng.choice([600, 900, 1500, 2400])
        return _Response("demo-thinker", _Usage(
            prompt_tokens=3000,                              # includes cached
            prompt_tokens_details={"cached_tokens": 2000},
            completion_tokens=thinking + 300,                # includes thinking
            completion_tokens_details=_OpenAIDetails(thinking)))


class _OtherClient:
    def __init__(self, rng):
        self.chat = type("chat", (), {"completions": _Completions(rng)})()


def run_api_demo(tasks=15):
    """Show the per-task report without needing an API key or any real spend."""
    configure(path=_start_clean("./qvunex_api_demo.jsonl"), sample_interval=0.5)
    rng = random.Random(11)
    client = wrap(_Client(rng))
    other = wrap(_OtherClient(rng))

    print(f"simulating {tasks} finished tasks (no network, no API key) ...")
    for i in range(tasks):
        with task("outbound email"):
            for _ in range(40):
                client.messages.create(model="demo-small",
                                       qvunex_route="classify intent")
            client.messages.create(model="demo-large", qvunex_route="research")
            if rng.random() < 0.15:
                # a retry you handled yourself: the same work, paid for twice
                record_retry("draft", dur_ms=820.0, attempt=2,
                             model="demo-large",
                             usage={"tokens_in": 1200, "tokens_out": 700})
            client.messages.create(model="demo-large", qvunex_route="draft")

    for _ in range(5):
        with task("support triage"):
            other.chat.completions.create(model="demo-thinker",
                                          qvunex_route="triage")

    flush()
    print()
    print(render(analyse(_config["path"], prices=parse_prices(DEMO_PRICES))))
    print(f"raw events: {_config['path']}")
    print("prices in this demo are invented. Copy tools/prices.example.txt to")
    print("~/.qvunex/prices.txt and put the published rates in it yourself.")
    return 0
