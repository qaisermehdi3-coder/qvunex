# Your config on a rented GPU: A100 80GB, and how it compares with an RTX 3090

Second run of [`planb.sh`](../../../planb.sh), 29 Oct 2026. Same model, same prompts and the same vLLM version as the [RTX 3090 run](../planb-3090-2026-10-27), plus one more load level (256 at once).

## Setup

- GPU: NVIDIA A100 80GB PCIe, 81920 MiB, driver 590.48.01 (vast.ai, machine m:56760, listed in the United Kingdom), $1.017/hour
- vLLM 0.27.1 (official image), default flags
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Load: prompt about 512 tokens, answer 256 tokens (`ignore_eos`), 1, 4, 16, 64 and 256 requests at once, 60 s each
- Meter: `qvunex_single.py` md5 5b49a0fa; script `planb.sh` md5 14f86e0a

## Results

- Start-up: 89.7 s from launch to `/health`, model already on disk. KV cache 56.3 GiB = 1,054,144 tokens
- 1 at once: 89 output tokens/s, p50 2.78 s, $3.161 per 1M output tokens
- 4 at once: 352 output tokens/s, p50 2.85 s, $0.803 per 1M
- 16 at once: 1,164 output tokens/s, p50 3.45 s, $0.243 per 1M
- 64 at once: 2,675 output tokens/s, p50 5.98 s / p95 7.11 s, $0.106 per 1M
- 256 at once: 3,433 output tokens/s, p50 18.71 s, $0.082 per 1M
- 0 errors at every level. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens, all five levels.

## Compared with the RTX 3090 ($0.259/hour)

At the same load, the 3090 is cheaper per token: at 64 at once $0.082 vs $0.106 per 1M, at 16 at once $0.163 vs $0.243.

At the same answer time, the A100 is cheaper or the only option:

- Answer in about 3 s: only the A100 can (3090 at 1 at once already takes 6.27 s).
- Answer in about 6 to 7 s: 3090 at 4 at once, p50 6.82 s / p95 7.19 s, $0.490 per 1M. A100 at 64 at once, p50 5.98 s / p95 7.11 s, $0.106 per 1M. The A100 is about 4.6 times cheaper per token.
- Answer in about 18 s: 3090 at 64 at once (17.87 s) and A100 at 256 at once (18.71 s) both cost $0.082 per 1M.

So which GPU is cheapest depends on how fast your answers must be and how many requests you can keep in flight. It also depends on the hourly prices: these are two vast.ai listings, and different prices change the answer.

## Limits

- Cost assumes the GPU is busy the whole window. Idle time in a real day raises it; only real traffic shows by how much.
- One run per level on each card, different hosts and days, synthetic prompts. Windows ran 61.1 to 66.9 s because in-flight requests finish, and that time is paid for, so it is counted.
- The 1-at-once window includes a one-time kernel compile (vLLM logged it at 14:59:47).
- The ERROR lines at the end of `server.log` (15:05:02) are vLLM shutting down after the last level, when the script stopped it.

## Files

- `report-planb.txt`: the report as printed
- the `planb-files` archive (tar.gz; open with `tar xzf`): server log, /metrics before and after each level, meter events, reconcile output
