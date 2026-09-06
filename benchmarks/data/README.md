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

**1. One run per config. The eager numbers are not reproducible; the CUDA graph
numbers are.**

Repeating fp16 batch 1 five times inside one container: CUDA graph timings spread
**0.23%**, eager timings spread **5.90%**. Running the same config again on a
*different* rented host a day later: graphs moved **+0.27%**, eager moved
**+42.1%**.

The derived claim "CUDA graphs are 1.33x faster" became "1.88x faster" on the
second host, from the same code and the same flags.

So a single constant per config is safe for the graph rows and quietly wrong for
the eager ones. If you are fitting coefficients, fit graphs and eager separately
and put an error bar on eager that a 42% host-to-host move would fit inside.

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
