# RTX 3090 — 24 GB, own measurement

Two CSVs, 96 timed runs, no failures. Measured 16 October 2026 by Qaiser Mehdi
on his own rented instance, not for a client. Free to use, free to argue with.

* `3090-own-1p5b.csv` — Qwen2.5-1.5B-Instruct and its AWQ build, 48 runs
* `3090-own-7b.csv` — Qwen2.5-7B-Instruct and its AWQ build, 48 runs
* `host-3090-own.txt` — what the container itself saw (GPU, driver, power limit, CPU, RAM, disk)

---

## Conditions

Same harness, flags and container image as the L40S run in `../l40s-own/`.
The exact script is `run3090.sh` in the repo root (commit 79cd34c).

| field | value |
|---|---|
| Engine | vLLM 0.27.1, pinned by container image `vllm/vllm-openai:v0.27.1` |
| torch | 2.13.0+cu130 |
| Driver | 580.159.03 |
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, power limit 360 W, max memory clock 9751 MHz |
| Batch sizes | 1, 8, 32, 128 |
| Modes | eager and CUDA graphs |
| Output tokens | 128 per request, `ignore_eos=True` |
| `max_model_len` | 1024 |
| `gpu_memory_utilization` | 0.90 |
| Prefix caching | disabled |
| Warmup | one untimed pass before each timed run |
| Process isolation | every configuration in a fresh subprocess |
| Repeats | 3 per configuration |

**Host.** vast.ai machine 75262, host 392144, the Netherlands. Listing:
AMD Ryzen Threadripper, **24.0 of 48 vCPUs allocated**, 32 of 64 GB RAM,
WD_BLACK NVMe at 5507 MB/s, PCIe 4.0 x16 at 25.5 GB/s, reliability 99.75%,
$0.414/hr. The container saw a Threadripper 7960X with 48 threads and 62 GB
(it sees the whole host, not the allocation). The `host_*_listed` columns in
the CSVs are empty for this run; the listing values are the ones written here.

**The limit, stated up front.** One card on one host. The L40S comparison below
is across two different hosts, with different CPUs, CPU allocations and drivers.

---

## What the data says

### 1. Against our L40S: even at small batch, behind at large batch

CUDA graphs, median of 3, L40S time divided by 3090 time (above 1.00x means the
3090 is faster).

| model | batch 1 | batch 8 | batch 32 | batch 128 |
|---|---|---|---|---|
| 1.5B fp16 | 1.09x | 0.99x | 0.92x | 0.70x |
| 1.5B AWQ | 1.01x | 0.96x | 0.79x | 0.50x |
| 7B fp16 | 1.03x | 1.04x | 1.11x | 0.72x |
| 7B AWQ | 1.09x | 1.01x | 0.71x | 0.46x |

Median of all 16: 0.98x. Range 0.46x to 1.11x.

At batch 1 and 8 the two cards are within 10% of each other on every model,
which is about what memory bandwidth alone predicts (936 vs 864 GB/s, 1.08x).
At batch 128 the 3090 is 0.46x to 0.72x. A single "3090 vs L40S" ratio hides
this: the answer depends on how many requests run at once.

### 2. Repeats: graphs and eager both repeat on this host

Spread = (slowest − fastest) / fastest of the 3 repeats, the same measure as the L40S notes.

| mode | mean spread | median | worst |
|---|---|---|---|
| CUDA graphs | 0.2% | 0.2% | 0.5% |
| eager | 1.5% | 0.9% | 7.1% |

0 of 48 eager runs and 0 of 48 graph runs came in more than 20% above their
own configuration's fastest run.

On the L40S host (8 of 64 vCPUs) eager spread was 37.3% on average and 12 of
48 eager runs were more than 20% slow. The L40S notes said the CPU-contention
explanation would be falsified by eager instability on a host with many
dedicated vCPUs. This host had 24 and eager was stable, which fits that
explanation. It does not prove it: the card is different too.

### 3. The precision flip shows up again

1.5B, batch 1:

* With CUDA graphs, AWQ is **45.2% faster** than fp16 (0.337 s vs 0.615 s).
* With eager, AWQ is **14.0% slower** than fp16 (1.420 s vs 1.246 s).

Same direction as on the L40S (49.0% faster vs 43.2% slower), smaller in
eager. Whether 4-bit is cheaper still depends on a flag most benchmarks do not
state.

### 4. Eager penalty on this card

| configuration | eager vs graphs (median of 3), batches 1 / 8 / 32 / 128 |
|---|---|
| 7B fp16 | +1.7% / +1.8% / +1.3% / +0.4% |
| 7B AWQ | +60.4% / +55.8% / +10.5% / +0.7% |
| 1.5B fp16 | +102.5% / +80.9% / +66.2% / +20.1% |
| 1.5B AWQ | +321.6% / +285.7% / +174.0% / +29.5% |

Same order as the L40S: 7B fp16 barely notices, 1.5B AWQ suffers most.

---

## What this does not cover

* **One model family.** Qwen2.5 only.
* **Short outputs.** 128 tokens per request.
* **One context cap.** `max_model_len` 1024.
* **Single GPU.** `tensor_parallel_size=1`.
* **No power or cost.** Time only.
* **One 3090.** Other 3090s can differ (cooling, power limit, host). This is one healthy-looking unit, not the model's average.
