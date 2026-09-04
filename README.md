# qvunex

**Measure what your AI inference actually costs.**

Most teams running AI in production know exactly one number: the monthly cloud
bill. They can't tell you cost per endpoint, per feature, or per prediction. When
the bill doubles, they can't say why.

This tells you.

```bash
pip install qvunex
```

```python
from qvunex import meter

@meter("checkout-classifier")
def predict(batch):
    return model(batch)
```

```bash
qvunex report
qvunex checklist
```

That's the whole integration.

---

## If you pay per token instead of per GPU-hour

Wrap the client and name the unit of work. Two lines.

```python
from qvunex import wrap, task

client = wrap(anthropic.Anthropic())      # or OpenAI

with task("outbound email"):
    ...your existing code, unchanged...
```

`wrap` matters more than it looks. Decorating your own functions only sees the
calls you wrote; wrapping the client also catches the calls your framework makes
on your behalf — which is where the spend hides when one request fans out into
six sub-agent calls.

Then give it a rate card (copy `tools/prices.example.txt` to
`~/.qvunex/prices.txt`) and run `qvunex report`:

```
--------------------------------------------------------------------
  COST PER FINISHED TASK
--------------------------------------------------------------------
  task                    n       mean        p50        p95   calls
  outbound email         12      $0.13      $0.13      $0.16    42.1
  support triage          5      $0.02      $0.02      $0.02     1.0

  retries                 1 call(s), $0.03   the same work paid for twice

--------------------------------------------------------------------
  COST BY ENDPOINT
--------------------------------------------------------------------
  endpoint                 calls   tokens $   share   per call
  classify intent            480      $1.04   62.0%    $0.0022
  research                    12      $0.27   16.2%      $0.02
  draft                       13      $0.27   16.0%      $0.02

--------------------------------------------------------------------
  WHAT THIS SUGGESTS
--------------------------------------------------------------------
  * 'classify intent' is 63.4% of token spend while being the cheapest
    single call you make ($0.0022). It is the 480 runs, not the price.
  * 'classify intent' wrote 408,000 tokens to the prompt cache and read
    none back. A write costs more than not caching at all, so that is a
    pure surcharge.
```

Four things there are hard to get any other way:

* **Cost per finished task**, not per request. Per-task cost cannot be
  reconstructed afterwards from per-call billing data — the id has to be attached
  at call time or the number is a guess.
* **The spread.** The mean and the p95 are different questions. An average hides
  the tasks that actually hurt.
* **Cache writes counted apart from reads.** A write costs *more* than not
  caching (1.25x input on the 5-minute TTL, 2x on the hour); a read costs 0.1x.
  Folded into one "input tokens" figure, a cold run and a warm run look the same.
* **Retries and fallbacks as their own line.** A retry is the same work paid for
  twice. A fallback is a model you didn't ask for answering, at its price. Both
  are real spend that logging the successful attempt drops on the floor.

Thinking tokens are recorded separately where the provider reports them, and are
*not* added to the bill again — they are a breakdown of tokens already counted.
A model with no entry in your rate card is reported as unpriced and left out
rather than estimated.

---

## What you get

```
====================================================================
  QVUNEX METER REPORT
====================================================================
  window        3600.0 s
  calls         41,209
  inferences    329,672
  gpu rate      $0.7500/hr x 1 device(s)
  window cost   $0.75
  CPKI          $0.0023 per 1,000 inferences

--------------------------------------------------------------------
  COST BY ENDPOINT
--------------------------------------------------------------------
  endpoint                    inf   share       cost        CPKI
  image-classifier        251,104   78.1%      $0.59     $0.0023
  text-embedder            62,336   18.6%      $0.14     $0.0022
  thumbnail-scorer         16,232    3.3%      $0.02     $0.0015

--------------------------------------------------------------------
  WASTE
--------------------------------------------------------------------
  mean GPU utilisation      31.4%
  time below 5% util        38.2%
  cost of that idle time    $0.29   <-- paid for, not used
  duty cycle                44.1%   time actually inside inference
  batch efficiency          19.0%   mean batch vs max seen (32)
```

Plus latency percentiles per endpoint, memory headroom, and a short list of
things worth looking at — fired only on defensible thresholds, because a meter
that cries wolf stops being read.

---

## The checklist

Two cost figures are only comparable if they were produced under the same conditions,
and those conditions almost never get published. `qvunex checklist` fills in what the
meter observed and prints everything else as **MISSING**, loudly:

```
04  WORKLOAD
------------------------------------------------------------------
  4.1  batch_size / concurrency   [observed]
        batch 1-32, mean 6.2 (6 distinct sizes)
  4.2  input_tokens / output_tokens
        >> MISSING - qvunex cannot observe this. State it yourself.
```

Tell it what it cannot see, and those fields fill in too:

```python
qvunex.configure(rate_usd_hour=0.35, context={
    "engine": "vLLM 0.27.1",
    "cuda_graphs": True,
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype": "fp16 weights, fp16 compute",
})
```

Why this is in the tool rather than in a document: I published a benchmark showing 4-bit
quantization was 24.8% *more* expensive than fp16, then re-ran it with CUDA graphs enabled
and measured 48.3% *cheaper*. One unreported flag, 73 percentage points, same card, same
afternoon. A checklist nobody fills in is not a standard.

Full field list and the evidence behind each one:
<https://gist.github.com/qaisermehdi3-coder/b00f296641681695daf90e5a500d0d23>

---

## Design choices, and why

**Local only.** Every measurement is appended to a JSONL file on your own disk.
This package contains no network code — grep it. That isn't a limitation, it's
the point: a read-only tool that never phones home gets adopted in an afternoon
instead of surviving a six-month security review.

**No dependencies.** Pure standard library. GPU stats come from shelling out to
`nvidia-smi`, which exists wherever an NVIDIA GPU does. Nothing for anyone's
platform team to approve.

**Never breaks the caller.** Any failure inside the meter is swallowed. Your
function's exceptions propagate untouched, and a call that raises is still
recorded with `ok=False`. A measurement tool that can take down production is a
tool nobody installs twice.

**Cheap.** The hot path is a `perf_counter` pair, a dict, and an append to a
buffered list. No I/O per call.

**Honest.** Costs are *attributed*, not measured: we know total GPU-hours and each
endpoint's share of busy time, and we divide. Under concurrency, summed call
duration exceeds wall time — so duty cycle is capped and the overlap is flagged
in the output rather than quietly normalised. Without a GPU price, cost is
reported as unavailable, never estimated from a default.

---

## Configuration

Set your GPU price to get costs. Everything else has a working default.

```python
import qvunex
qvunex.configure(
    path="~/.qvunex/events.jsonl",   # where the corpus lives
    rate_usd_hour=0.75,              # what you pay per GPU-hour
    sample_interval=1.0,             # device polling, seconds
)
```

Or by environment, which is usually easier in a container:

```
QVUNEX_PATH=/data/qvunex.jsonl
QVUNEX_RATE_USD_HOUR=0.75
QVUNEX_DISABLED=1          # hard off, decorator becomes a passthrough
```

For code that isn't shaped like a function:

```python
with meter.span("batch-job", batch=len(items)):
    process(items)
```

Batch size is inferred from the first argument (`.shape[0]`, then `len()`,
then 1). Override it when that guess is wrong:

```python
@meter("ranker", batch=lambda args, kwargs: len(kwargs["docs"]))
def rank(*, query, docs): ...
```

---

## CLI

```bash
qvunex report                        # default corpus
qvunex report /data/events.jsonl     # a specific file
qvunex report --rate 0.75            # override the recorded price
qvunex report --json                 # raw analysis for your own tooling
qvunex checklist                     # comparability fields, gaps marked MISSING
qvunex demo                          # synthetic workload + report, no setup
```

---

## The corpus

Events are JSONL, one object per line, schema-versioned in `schema.py`. Three
record types: `session`, `call`, `gpu`.

Keep these files. The corpus is the asset — cross-workload measurement data can't
be collected retroactively, and no cloud vendor can assemble it, because no cloud
vendor is neutral enough to be allowed to measure its competitors.

---

## Status

v0.2. Measures what it claims to measure and nothing more. Precision headroom and
input-triviality analysis are deliberately not here yet — those change your model,
and this release only observes.

`qvunex checklist` exits non-zero when fields are missing, so you can wire it into
CI and fail a build that would publish an uncomparable number.

Apache-2.0.
