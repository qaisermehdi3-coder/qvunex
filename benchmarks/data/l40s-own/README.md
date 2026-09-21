# L40S — 48 GB, own measurement

Two CSVs, 96 timed runs, no failures. Measured 21 September 2026 by Kaisar Mehdi
on his own rented instance, not for a client. Free to use, free to argue with.

* `l40s-own-1p5b.csv` — Qwen2.5-1.5B-Instruct and its AWQ build, 48 runs
* `l40s-own-7b.csv` — Qwen2.5-7B-Instruct and its AWQ build, 48 runs

---

## Conditions

Every row carries its own conditions. These were held fixed across all 96 runs.

| field | value |
|---|---|
| Engine | vLLM 0.27.1, pinned by container image `vllm/vllm-openai:v0.27.1` |
| torch | 2.13.0+cu130 |
| Driver | 595.84 |
| GPU | NVIDIA L40S, 46068 MiB reported by `nvidia-smi` |
| Batch sizes | 1, 8, 32, 128 |
| Modes | eager and CUDA graphs |
| Output tokens | 128 per request, `ignore_eos=True` so every request generates exactly 128 |
| `max_model_len` | 1024 |
| `gpu_memory_utilization` | 0.90 |
| `tensor_parallel_size` | 1 |
| Prefix caching | disabled |
| Prompt prefixes | unique per request |
| Warmup | one untimed pass before each timed run |
| Process isolation | every configuration in a fresh subprocess |
| Repeats | 3 per configuration |
| Seed | 0 |

**Host.** vast.ai machine 150627, host 640216, Romania. Intel Xeon Silver 4514Y,
**8.0 of 64 vCPUs allocated**, 64 GB RAM, Samsung MZ7L37T6 SATA SSD at
2947 MB/s — not NVMe. Host reliability 97.2%. Host fields are the provider's
listing; the container's own probe is recorded beside them in every row so a
disagreement is visible.

**The limit, stated up front.** This is the same physical machine as an earlier
L40S measurement made for a client. Only one L40S was listed on the market on
both occasions. So this is a card measured on my own account and mine to
publish — it is **not** an independent second host, and nothing here is a
host-to-host replication.

---

## What the data says

### 1. Three repeats, one machine, one session: graphs repeat, eager does not

Same container, same process-isolated harness, three repeats per configuration,
one after another.

| mode | mean spread | median | worst |
|---|---|---|---|
| CUDA graphs | 0.7% | 0.4% | 2.6% |
| eager | 37.3% | 42.1% | 78.5% |

Put another way:

**12 of 48 eager runs came in more than 20% above their own configuration's
fastest run. 0 of 48 graph runs did.**

This is not host-to-host variance. It is the same box, minutes apart. A single
eager timing is not a property of the card and is not even a property of the
machine — it is a property of that minute.

The slow runs arrive sporadically rather than drifting, which is what sharing
8 of 64 vCPUs with other tenants looks like.

### 2. Eager's penalty and eager's instability are the same thing

The exception is the interesting part. **7B fp16 in eager mode is stable** —
0.1%, 0.1%, 0.3%, 0.6% across the four batch sizes. It is the only eager
configuration here that repeats.

It is also the only one where eager costs almost nothing:

| configuration | eager vs graphs (median of 3) |
|---|---|
| 7B fp16, batches 1 / 8 / 32 / 128 | +1.1% / +1.5% / +1.9% / +1.5% |
| 7B AWQ | +54.9% / +43.3% / +102.5% / +27.3% |
| 1.5B fp16 | +65.6% / +57.3% / +51.6% / +121.2% |
| 1.5B AWQ | +365.0% / +333.1% / +214.9% / +141.3% |

The two effects line up exactly. Where eager is cheap it is also stable; where
eager is expensive it is also unstable.

The working explanation: eager dispatches every operation from Python, so each
step carries a fixed CPU cost. When the GPU work per step is large — 7B at
fp16 — that cost is hidden behind the GPU and a CPU hiccup does not show.
When the GPU work per step is small — a 1.5B model, or a 4-bit build that
finishes quickly — Python dispatch becomes the bottleneck, and the host's CPU
contention lands directly in the number.

**The practical rule:** eager is only safe to benchmark with when the model is
big enough and the precision heavy enough that the GPU is the bottleneck. On
this card that was 7B fp16 and nothing else tested.

*This is an explanation consistent with the data, not a proven mechanism.* It
would be falsified by a run showing the same eager instability on a host with
many dedicated vCPUs.

### 3. One flag reverses which precision is cheaper

1.5B, batch 1, same card, same session, forty minutes apart:

* With CUDA graphs, the AWQ build is **49.0% faster** than fp16.
* With eager, the AWQ build is **43.2% slower** than fp16.

A swing of **92 percentage points** on a flag that most published benchmarks do
not state. The direction of the answer — "is 4-bit cheaper here" — depends on
it.

At 7B the flip does not happen; AWQ wins in both modes. So this is not a
provider constant either.

### 4. Worst-to-best spread: 423.7x

Slowest configuration, 7B fp16 eager at batch 1: 2.587 s per inference.
Fastest, 1.5B AWQ with graphs at batch 128: 0.006105 s per inference.

Medians of three. A spread quoted from a single run would describe one minute
on one machine.

---

## What this does not cover

* **One model family.** Qwen2.5 only.
* **Short outputs.** 128 tokens per request. Do not extrapolate to long generation.
* **One context cap.** `max_model_len` 1024. The card's natural ceiling was not measured.
* **Single GPU.** `tensor_parallel_size=1` throughout.
* **No power measurement.** Measured for time, not watts.
* **No cost.** Convert these timings to money with your own rate if you like;
  nothing in these files is a price.
* **One host.** See the limit stated above.

---

## Reproducing

The harness is one file:

<https://github.com/qaisermehdi3-coder/qvunex/blob/main/benchmarks/sweep.py>

It runs each configuration in a fresh subprocess, disables prefix caching, gives
every prompt a unique prefix, pins `ignore_eos` with fixed `max_tokens`, reads
the GPU name from `nvidia-smi` rather than torch, records a failed configuration
as a row rather than losing the sweep, and takes `--repeats N` so a configuration
is measured against itself.

The exact script used for this run, host flags and all, is `run.sh` in the repo
root.

If you run it on hardware not covered here, the CSV is welcome.
