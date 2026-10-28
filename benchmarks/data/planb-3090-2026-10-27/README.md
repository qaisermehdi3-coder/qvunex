# Your config on a rented GPU: sample report (RTX 3090, vLLM 0.27.1)

First full run of [`planb.sh`](../../../planb.sh), 27 Oct 2026. It is for teams who will not run our script on their own servers: we rent the same GPU, run their model and settings, and send them this report.

## Setup

- GPU: NVIDIA GeForce RTX 3090, 24576 MiB, driver 580.159.03 (vast.ai, host 443829, California), $0.259/hour
- vLLM 0.27.1 (official image), default flags
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Load: prompt about 512 tokens, answer 256 tokens (`ignore_eos`), 1, 4, 16 and 64 requests at once, 60 s each
- Meter: `qvunex_single.py` md5 5b49a0fa; script `planb.sh` md5 14f86e0a

## Results

- Start-up: 122.9 s from launch to `/health`, model already on disk
- 1 at once: 40 output tokens/s, p50 6.27 s, $1.809 per 1M output tokens
- 4 at once: 147 output tokens/s, p50 6.82 s, $0.490 per 1M
- 16 at once: 442 output tokens/s, p50 9.06 s, $0.163 per 1M
- 64 at once: 878 output tokens/s, p50 17.87 s, $0.082 per 1M
- 0 errors at every level. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens, all four levels.

The same card costs 22 times less per output token at 64 at once than at 1, and each answer takes about 2.9 times longer.

## Limits

- Cost assumes the GPU is busy the whole window. Idle time in a real day raises it; only real traffic shows by how much.
- One card, one run per level, synthetic prompts. Each window runs until in-flight requests finish (62.8 to 71.7 s), and that time is paid for, so it is counted.
- The 1-at-once window includes a one-time kernel compile (vLLM logged it at 18:56:34).

## Files

- `report-planb.txt`: the report as printed
- `planb-files.tgz.gz`: server log, /metrics before and after each level, meter events, reconcile output (a tar.gz archive; open with `tar xzf`)
