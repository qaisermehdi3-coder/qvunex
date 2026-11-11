# RTX 5090: one long prompt among short streamed answers

First run of [`mixed.sh`](../../../mixed.sh), 9 Nov 2026 (UTC). It tests what one long prompt does to the short answers already being streamed, with vLLM's default chunked prefill and with a batch budget large enough to take the whole long prompt in one step.

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04 (vast.ai, machine m:149075, listed in South Korea, datacenter 406325), $0.522/hour
- Host: AMD Ryzen 7 7800X3D, 16 threads, B850 EAGLE ICE, the same build as the earlier RTX 5090 runs in this datacenter ([m:143238](../planb-5090-2026-11-07), [m:144985](../planb-5090-prompt2000-2026-11-08))
- vLLM 0.27.1 (official image), model `Qwen/Qwen2.5-7B-Instruct`
- Load, 60 s per phase, every answer streamed, 256 tokens (`ignore_eos`):
  - short only: 16 requests at once, prompts of about 512 tokens
  - mixed: the same 16, plus one request with a prompt of about 12,000 tokens every 10 s (6 per phase)
- Two server configs, restarted between them:
  1. default flags (chunked prefill on)
  2. `--max-num-batched-tokens 16384`: still chunked prefill, but the budget is large enough for the whole 12,000-token prompt in one step, so it is not split
- Measured per request from the client: whole answer time, first token, and the biggest pause between two streamed tokens inside one answer
- Meter: `qvunex_single.py` md5 5b49a0fa; script `mixed.sh` md5 6f42ad54

## Results

Short answers, mixed phase (16 short at once plus the long prompts):

- Biggest pause inside an answer, p95: 0.180 s with default flags vs 0.973 s with the 16,384 budget (5.4x)
- Short answers with a pause over 0.3 s: 0 of 240 vs 105 of 240
- Whole answer: p50 4.01 s / p95 4.79 s vs p50 3.92 s / p95 4.99 s
- First token: p50 0.334 s / p95 0.478 s vs p50 0.575 s / p95 0.748 s

Long prompts (about 12,000 tokens), first token: 0.992 to 1.001 s with default flags vs 1.004 to 1.010 s with the 16,384 budget. Whole answer p50 5.06 s vs 5.08 s.

Cost per 1M output tokens in the mixed phase: $0.141 vs $0.142. Output: 1,031 vs 1,023 tokens/s.

Short only, for reference (no long prompts):

- Default flags: whole answer p50 3.38 s / p95 3.90 s, first token p50 0.331 s, biggest pause p95 0.134 s, $0.122 per 1M output tokens
- 16,384 budget: whole answer p50 3.40 s / p95 3.74 s, first token p50 0.570 s, biggest pause p95 0.528 s (18 of 288 answers over 0.3 s), $0.122 per 1M

0 errors in every phase. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens in all four phases. Every request returned 256 output tokens.

## What it shows

With the whole long prompt in one step, the short answers already streaming stopped for about a second each time one arrived: the p95 pause (0.973 s) is almost the long prompt's whole first-token time (about 1.0 s). With default flags the same prefill was split, and the longest pause any short answer saw was 0.181 s.

The long request did not pay for the split here: its first token came at about 1.0 s both ways. Cost per output token was the same both ways. The gain from chunking in this run was smoother streams for everyone else, at no cost that showed up.

Adding the long prompts with default flags raised the short answers' p95 from 3.90 s to 4.79 s and the cost per 1M output tokens from $0.122 to $0.141. The card spent part of each minute reading about 72,000 extra prompt tokens, and reading prompts produces no output tokens.

## The default batch budget

vLLM did not log its batch budget for the default config (it did for config 2: "Chunked prefill is enabled with max_num_batched_tokens=16384"). In the vLLM 0.27.1 source (`vllm/engine/arg_utils.py`, `get_batch_defaults`), the API server's default on a GPU under 70 GB is 2,048 tokens. The run fits that: a 12,000-token prompt in chunks of about 2,000 is about 6 steps, and the longest pause in the default mixed phase (0.181 s) is about one sixth of the long prompt's first-token time (about 1.0 s). This is consistent with 2,048, not a log of it.

## Limits

- One run per phase. Six long prompts per mixed phase, at a fixed 10 s interval. Real traffic arrives in bursts and mixes many lengths.
- One card, one model, synthetic prompts (unique prefix, then a repeated word).
- No config with chunked prefill turned off. `--max-num-batched-tokens 16384` stands in for it at this prompt length; in vLLM 0.27.1, turning chunked prefill off for this model logs that it "may cause the engine to crash".
- Start-up: 142.9 s for the first server, 45.3 s for the second. The second start ran on a machine that had just started the same model (weights load 1.9 s vs 7.8 s, graph capture 4 s vs 58 s), so it is not a fair start-up comparison.
- vLLM logged a one-time Triton kernel compile at the first requests of the first phase (10:15:01).
- Costs assume the GPU is busy the whole window.

## Files

- the `mixed-files` archive (tar.gz; open with `tar xzf`): both server logs, /metrics before and after each phase, per-request timings (`requests_*.jsonl`), meter events, reconcile output, the report, and the exact `mixed.sh` and meter that ran
