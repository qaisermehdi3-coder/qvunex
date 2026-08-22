# qvunex

**Measure what your AI inference actually costs.**

Most teams running AI in production know exactly one number: the monthly cloud
bill. They can't tell you cost per endpoint, per feature, or per prediction. When
the bill doubles, they can't say why.

This tells you.

```bash
pip install git+https://github.com/qaisermehdi3-coder/qvunex
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
