# Your config on a rented GPU: H100 80GB SXM, compared with the RTX 5090, A100 and RTX 3090

Fourth run of [`planb.sh`](../../../planb.sh), 7 Nov 2026. Same model, prompts and vLLM version as the [RTX 3090](../planb-3090-2026-10-27), [A100](../planb-a100-2026-10-29) and [RTX 5090](../planb-5090-2026-11-07) runs, with the 256-at-once level the A100 also ran.

## Setup

- GPU: NVIDIA H100 80GB HBM3 (SXM), 81559 MiB, driver 615.71.09 (vast.ai, machine m:140932, listed in Czechia), $3.557/hour
- CPU: Intel Xeon Platinum 8568Y+, 192 threads
- vLLM 0.27.1 (official image), default flags
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Load: prompt about 512 tokens, answer 256 tokens (`ignore_eos`), 1, 4, 16, 64 and 256 requests at once, 60 s each
- Meter: `qvunex_single.py` md5 5b49a0fa; script `planb.sh` md5 3c69b8cc

## Results

- Start-up: 70.8 s from launch to `/health`, model already on disk (the download, 64 s, is not counted). KV cache 55.61 GiB = 1,041,168 tokens
- 1 at once: 163 output tokens/s, p50 1.54 s / p95 1.55 s, $6.079 per 1M output tokens
- 4 at once: 630 output tokens/s, p50 1.59 s / p95 1.60 s, $1.569 per 1M
- 16 at once: 2,239 output tokens/s, p50 1.79 s / p95 1.83 s, $0.441 per 1M
- 64 at once: 6,102 output tokens/s, p50 2.58 s / p95 2.83 s, $0.162 per 1M
- 256 at once: 9,994 output tokens/s, p50 6.23 s / p95 7.55 s, $0.099 per 1M
- 0 errors at every level. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens, all five levels.

## Compared with the other three cards

The H100 was the fastest card at every load, and the most expensive per hour of the four ($3.557, against $1.017 for the A100, $0.522 for the RTX 5090 and $0.259 for the RTX 3090). Which card is cheapest per token depends on the answer time:

- p95 under 2 s: only the H100 was fast enough. 16 at once, $0.441 per 1M.
- p95 under 3 s: H100 at 64 at once, $0.162; RTX 5090 at 4 at once, $0.382; A100 at 4 at once, $0.803. The H100 is about 2.4x cheaper than the 5090.
- p95 under 4 s: RTX 5090 at 16 at once, $0.120; H100 at 64 at once, $0.162. The 5090 is about 1.4x cheaper.
- p95 under 8 s: RTX 5090 at 64 at once, $0.050; H100 at 256 at once, $0.099; A100 at 64 at once, $0.106; RTX 3090 at 4 at once, $0.490. The 5090 is about 2x cheaper than the H100.

## Limits

- Cost assumes the GPU is busy the whole window. Idle time in a real day raises it; only real traffic shows by how much.
- $3.557/hour is one vast.ai listing on one day. Cost scales with price: at $1.78/hour the same H100 numbers would be half. Re-price it with `cheapest.py --price "H100=..."` or on the site.
- 256 at once was the highest level here. The H100 still answered within 8 s (p95) at 256, so its cost at longer answer times was not measured.
- One run per level, a different host from the other runs, synthetic prompts.
- vLLM logged a one-time kernel compile at the first request of the first level (18:10:24). The WARNING lines at the end of `server.log` (18:15:37) are vLLM shutting down after the last level, when the script stopped it.

## Files

- the `planb-files` archive (tar.gz; open with `tar xzf`): server log, /metrics before and after each level, meter events, reconcile output, the report, `corpus-rows.csv`, and the exact `planb.sh` and meter that ran
