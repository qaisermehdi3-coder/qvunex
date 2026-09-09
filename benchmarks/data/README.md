# Measured data

Two sweeps, both run by hand on rented hardware. Raw numbers, with the caveats
that make them safe to reuse. If you are calibrating a simulator or a cost model
against these, read the caveats first — one of them is large enough to change a
conclusion.

## Files

### `l4-verified.csv` — NVIDIA L4, 16 configs

Produced by `benchmarks/sweep.py` on an L4 rented through Modal, September 2026.

| column | meaning |
|---|---|
| `gpu` | device name, read from `nvidia-smi` |
| `format` | `fp16` or `awq` (4-bit) |
| `graph_mode` | `graphs` (CUDA graphs) or `eager` |
| `batch` | requests in the batch |
| `seconds_total` | wall time for the whole batch |
| `seconds_per_inference` | `seconds_total / batch` |

Model: Qwen2.5-1.5B-Instruct and its AWQ build. vLLM 0.27.1, 128 output tokens,
`ignore_eos=True` so every request generates exactly that many, `max_model_len=1024`.

**Prefix caching was disabled and every prompt was given a unique prefix.** Each
config ran in a fresh subprocess after a warmup pass. This matters: an earlier
version of this sweep left prefix caching on and had no warmup, and it reported a
worst-to-best spread of 284x. The corrected spread is **137x**. The directions of
every finding held; the magnitudes did not.

### `l4-run2.csv` — NVIDIA L4 again, different host, eight days later

Same script, same models, same flags, same card model. A different rented
machine. Produced 2026-09-09 while rehearsing a two-model-size run; the 7B half
of that rehearsal failed and is not included here.

This file exists so the reproducibility claim below can be checked rather than
taken on trust. Column layout differs slightly from `l4-verified.csv` because it
was produced by a later version of `sweep.py` which also records `repeat` and
host CPU; the measurement columns are the same.

### `t4-006.csv` — NVIDIA Tesla T4, 16 configs, with power

Google Colab T4, August 2026, same model pair and settings. Adds board power and
derived energy, which the L4 run does not have.

| column | meaning |
|---|---|
| `config` | `eager` or `cudagraphs` |
| `model` | `fp16` or `awq-int4` |
| `batch` | requests in the batch |
| `tok_per_sec` | throughput |
| `watts` | mean board power over the run, from `nvidia-smi` |
| `usd_per_1m_tokens` | at $0.35/GPU-hour, the rate stated in the header |
| `joules_per_token` | derived from watts and throughput |

Header comment lines begin with `#`. Skip them when parsing.

## Caveats — read these before calibrating anything

**1. One run per config. The eager numbers are not reproducible; the fp16 CUDA
graph numbers are.**

`l4-verified.csv` and `l4-run2.csv` are the same 16 configs on the same card
model, eight days and one rented host apart. Comparing them config by config:

| config | 1 Sept | 9 Sept | change |
|---|---|---|---|
| fp16 graphs, batch 1 | 1.710529 | 1.700665 | **-0.6%** |
| fp16 graphs, batch 8 | 0.220472 | 0.221636 | **+0.5%** |
| fp16 graphs, batch 32 | 0.065575 | 0.064882 | **-1.1%** |
| fp16 graphs, batch 128 | 0.023348 | 0.023653 | **+1.3%** |
| fp16 eager, batch 1 | 2.266555 | 3.383616 | **+49.3%** |
| fp16 eager, batch 8 | 0.335448 | 0.462567 | **+37.9%** |
| fp16 eager, batch 32 | 0.079215 | 0.119232 | **+50.5%** |
| fp16 eager, batch 128 | 0.024361 | 0.034148 | **+40.2%** |

Every fp16 graph config landed within 1.3%. Every fp16 eager config moved between
38% and 51%. Across all sixteen configs the mean absolute change was 7.1% for
graphs and 42.5% for eager.

**AWQ with graphs is the exception and we are not going to pretend otherwise.**
It moved +2.4%, +2.9%, +23.7% and +24.2% on the four batch sizes. So
"graphs are reproducible" holds cleanly for fp16 and only partly for AWQ, and
anyone calibrating an AWQ config from a single run should not assume the 1%
figure applies to them.

**The headline number moves too.** Worst-to-best spread was 137.5x on the first
host and 203.4x on the second. If you quote a spread from one run, you are
quoting a property of that machine as though it were a property of the card.

So a single constant per config is safe for fp16 graph rows and quietly wrong for
eager ones. If you are fitting coefficients, fit graphs and eager separately, and
put an error bar on eager wide enough for a 50% host-to-host move.

Working hypothesis, unverified: eager dispatches every operation from Python, so
it tracks the host CPU; CUDA graphs replay a fixed schedule and barely touch it.
If that is right, host CPU is a comparability field that benchmarks do not record.
`sweep.py` v2 records CPU model and core count in every row for this reason.

**2. Cost columns are a rate times a time, not a measurement.** `usd_per_1m_tokens`
is throughput against a stated $0.35/GPU-hour. Substitute your own rate.

**3. Small model, short outputs.** A 1.5B model generating 128 tokens. Do not
extrapolate the ratios to a 70B model or to long-context work without checking.

**4. The T4 file has power; the L4 file does not.** Different rigs, different
access. Nothing is inferred across them.

## Reproducing

```bash
python benchmarks/sweep.py --help
```

Single file, runs each config in a fresh subprocess, records a failed config
rather than losing the sweep, and takes `--repeats N` to print how much each
config disagrees with itself. If you run it on hardware not listed here, the CSV
is welcome — that is the whole point of it being one file.
