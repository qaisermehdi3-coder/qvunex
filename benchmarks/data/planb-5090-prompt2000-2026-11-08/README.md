# RTX 5090 with 2,000-token prompts: what longer prompts cost

Sixth run of [`planb.sh`](../../../planb.sh), 8 Nov 2026 (UTC). The same card, model, answer length and vLLM as the [first RTX 5090 run](../planb-5090-2026-11-07), on a twin machine in the same datacenter, with one change: prompts of about 2,000 tokens instead of about 512.

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04 (vast.ai, machine m:144985, listed in South Korea, datacenter 406325), $0.522/hour
- The first RTX 5090 run used m:143238 in the same datacenter: same board (B850 EAGLE ICE), CPU (AMD Ryzen 7 7800X3D, 16 threads), PCIe 5.0, driver and price. Not the same machine, but the same build.
- vLLM 0.27.1 (official image), default flags (chunked prefill on, as vLLM 0.27.1 does by default)
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Load: prompt about 2,000 tokens (2,031 to 2,032 measured per request), answer 256 tokens (`ignore_eos`), 1, 4, 16 and 64 requests at once, 60 s each
- Meter: `qvunex_single.py` md5 5b49a0fa; script `planb.sh` md5 84a359aa (v6, its first run on a real GPU)

## Results

- Start-up: 141.9 s from launch to `/health`, model already on disk. KV cache 12.21 GiB = 228,576 tokens
- 1 at once: 96 output tokens/s, p50 2.62 s / p95 2.62 s, first token 0.136 s, $1.510 per 1M output tokens
- 4 at once: 325 output tokens/s, p50 3.12 s / p95 3.27 s, first token 0.306 s, $0.446 per 1M
- 16 at once: 795 output tokens/s, p50 5.09 s / p95 5.79 s, first token 0.446 s, $0.182 per 1M
- 64 at once: 1,211 output tokens/s, p50 13.29 s / p95 20.00 s, first token 1.474 s, $0.120 per 1M
- 0 errors at every level. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens, all four levels. First token is the mean from vLLM's own histogram; its count equals the requests done at every level.

## Same card, prompts of about 512 vs about 2,000 tokens

Same answer length (256), same price ($0.522/hour):

- 1 at once: $1.441 vs $1.510 per 1M output tokens (+5%); answer p50 2.50 s vs 2.62 s; first token 0.047 s vs 0.136 s
- 4 at once: $0.382 vs $0.446 (+17%); 2.67 s vs 3.12 s; first token 0.119 s vs 0.306 s
- 16 at once: $0.120 vs $0.182 (+52%); 3.36 s vs 5.09 s; first token 0.276 s vs 0.446 s
- 64 at once: $0.050 vs $0.120 (2.4x); 5.41 s vs 13.29 s; first token 0.467 s vs 1.474 s

With one request at a time, four times longer prompts cost about 5% more per output token. With 64 in flight they cost 2.4 times more, and answers take 2.5 times longer: at high load the card spends much of its time reading prompts, which produces no output tokens. Cost per output token is the unit here, so this is where the prompt length shows.

Start-up matched the first host almost exactly: 141.9 s against 141.9 s, CUDA graph capture 60 s against 59 s, torch.compile 8.5 s against 8.6 s. On the [Norway host](../planb-5090-2026-11-08) with a different CPU the same steps took 116 s and 17.3 s. This twin supports the reading that the start-up gap there came from the host, not from chance.

## Limits

- Cost assumes the GPU is busy the whole window. Idle time in a real day raises it; only real traffic shows by how much.
- One run per level, synthetic prompts (unique prefix, then a repeated word), every prompt the same length. Real traffic mixes lengths; this run does not test long prompts arriving among short ones.
- 256 at once was not run: 256 prompts of about 2,300 tokens each would need more KV cache than this card has (228,576 tokens).
- vLLM logged a one-time kernel compile at the first request of the first level (17:09:34). The WARNING lines at the end of `server.log` (17:13:44) are vLLM shutting down after the last level, when the script stopped it.

## Files

- the `planb-files` archive (tar.gz; open with `tar xzf`): server log, /metrics before and after each level, meter events, reconcile output, the report, `corpus-rows.csv`, and the exact `planb.sh` and meter that ran
