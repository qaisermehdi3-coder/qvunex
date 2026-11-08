# RTX 5090 again, on a second host: how much does the host change the numbers?

Fifth run of [`planb.sh`](../../../planb.sh), 8 Nov 2026. The same card, model, prompts and vLLM as the [first RTX 5090 run](../planb-5090-2026-11-07), on a different machine in a different country, plus the 256-at-once level the first run did not have.

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 595.71.05 (vast.ai, machine m:149848, listed in Norway), $0.762/hour
- CPU: AMD EPYC 7B13, PCIe 4.0 slot (the first host: AMD Ryzen 7 7800X3D, PCIe 5.0)
- vLLM 0.27.1 (official image), default flags
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Load: prompt about 512 tokens, answer 256 tokens (`ignore_eos`), 1, 4, 16, 64 and 256 requests at once, 60 s each
- Meter: `qvunex_single.py` md5 5b49a0fa; script `planb.sh` md5 9a33bfb3 (v5, its first run on a real GPU)

## Results

- Start-up: 249.2 s from launch to `/health`, model already on disk (the download, 83 s, is not counted). KV cache 12.18 GiB = 228,128 tokens
- 1 at once: 99 output tokens/s, p50 2.50 s / p95 2.51 s, $2.133 per 1M output tokens
- 4 at once: 380 output tokens/s, p50 2.64 s / p95 2.65 s, $0.557 per 1M
- 16 at once: 1,209 output tokens/s, p50 3.32 s / p95 3.69 s, $0.175 per 1M
- 64 at once: 3,026 output tokens/s, p50 5.26 s / p95 6.00 s, $0.070 per 1M
- 256 at once: 3,781 output tokens/s, p50 17.18 s / p95 24.27 s, $0.056 per 1M
- 0 errors at every level. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens, all five levels.

## Same card, two hosts

Output tokens per second, first host (South Korea) vs this one (Norway):

- 1 at once: 101 vs 99
- 4 at once: 379 vs 380
- 16 at once: 1,208 vs 1,209
- 64 at once: 2,901 vs 3,026 (4.3% more here)

Cost per 1M output tokens at the same price ($0.522/hour): 1.441 vs 1.461, 0.382 vs 0.382, 0.120 vs 0.120, 0.050 vs 0.048. Serving speed moved by at most about 4% between the two machines.

Start-up did not: 249.2 s here against 141.9 s on the first host. vLLM's own log shows where: CUDA graph capture took 116 s here against 59 s, and torch.compile 17.3 s against 8.6 s, while loading the weights was faster here (2.4 s against 7.0 s). Graph capture and compile are CPU-heavy steps, so the slower CPU is the likely reason; this run does not prove it. For a server that scales to zero, the host can roughly double the paid start-up of every wake.

## What the 256 level adds

At 256 at once the 5090 answered with p95 24.27 s and cost $0.056 per 1M output tokens at $0.762/hour, or about $0.038 at the first host's $0.522. The first run stopped at 64 at once, so this is the lowest measured 5090 cost when answers may take about 25 s.

## Limits

- Cost assumes the GPU is busy the whole window. Idle time in a real day raises it; only real traffic shows by how much.
- Two hosts are not a distribution. They show that host-to-host spread exists and roughly how large it was here, not what it is in general.
- One run per level on each host, synthetic prompts, one model.
- vLLM logged a one-time kernel compile at the first request of the first level (07:47:53). The WARNING lines at the end of `server.log` (07:53:09) are vLLM shutting down after the last level, when the script stopped it.

## Files

- the `planb-files` archive (tar.gz; open with `tar xzf`): server log, /metrics before and after each level, meter events, reconcile output, the report, `corpus-rows.csv`, and the exact `planb.sh` and meter that ran
