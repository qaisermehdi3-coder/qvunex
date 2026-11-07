# Your config on a rented GPU: RTX 5090, compared with the A100 and RTX 3090

Third run of [`planb.sh`](../../../planb.sh), 7 Nov 2026. Same model, prompts and vLLM version as the [RTX 3090 run](../planb-3090-2026-10-27) and the [A100 run](../planb-a100-2026-10-29).

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04 (vast.ai, machine m:143238, listed in South Korea), $0.522/hour
- CPU: AMD Ryzen 7 7800X3D, 16 threads
- vLLM 0.27.1 (official image), default flags
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Load: prompt about 512 tokens, answer 256 tokens (`ignore_eos`), 1, 4, 16 and 64 requests at once, 60 s each
- Meter: `qvunex_single.py` md5 5b49a0fa; script `planb.sh` md5 3c69b8cc

## Results

- Start-up: 141.9 s from launch to `/health`, model already on disk (the download, 202 s, is not counted). KV cache 12.21 GiB = 228,576 tokens
- 1 at once: 101 output tokens/s, p50 2.50 s / p95 2.51 s, $1.441 per 1M output tokens
- 4 at once: 379 output tokens/s, p50 2.67 s / p95 2.67 s, $0.382 per 1M
- 16 at once: 1,208 output tokens/s, p50 3.36 s / p95 3.50 s, $0.120 per 1M
- 64 at once: 2,901 output tokens/s, p50 5.41 s / p95 6.18 s, $0.050 per 1M
- 0 errors at every level. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens, all four levels.

## Compared with the A100 ($1.017/hour) and RTX 3090 ($0.259/hour)

At the same load, the 5090 was the cheapest per token at every level run on all three cards:

- 1 at once: 5090 $1.441, 3090 $1.809, A100 $3.161 per 1M
- 16 at once: 5090 $0.120, 3090 $0.163, A100 $0.243
- 64 at once: 5090 $0.050, 3090 $0.082, A100 $0.106

It was also slightly faster than the A100 at the same load (64 at once: 2,901 vs 2,675 output tokens/s) at about half the hourly price.

At the same answer time:

- p95 under 4 s: 5090 at 16 at once, $0.120 per 1M; A100 at 16 at once, $0.243. About 2x cheaper. The 3090 never answered this fast.
- p95 under 8 s: 5090 at 64 at once (p95 6.18 s), $0.050; A100 at 64 at once (p95 7.11 s), $0.106; 3090 at 4 at once (p95 7.19 s), $0.490. About 2.1x cheaper than the A100.

## Limits

- Cost assumes the GPU is busy the whole window. Idle time in a real day raises it; only real traffic shows by how much.
- 64 at once was the highest level on this card; the A100 also ran 256. So how low the 5090's cost goes when answers may take longer than about 6 s was not measured.
- 32 GB of memory: the KV cache holds 228,576 tokens, against 1,054,144 on the A100 80GB. Longer prompts or more requests at once use more of it; this run does not test where it runs out.
- One run per level, a different host and day from the other two runs, synthetic prompts, one vast.ai price. Different prices change the answer.
- vLLM logged a one-time kernel compile at the first request of the first level (12:52:31). The WARNING lines at the end of `server.log` (12:56:38) are vLLM shutting down after the last level, when the script stopped it.

## Files

- the `planb-files` archive (tar.gz; open with `tar xzf`): server log, /metrics before and after each level, meter events, reconcile output, the report, `corpus-rows.csv`, and the exact `planb.sh` and meter that ran
