# RTX 5090: bursts of long prompts among short streamed answers

Second run of [`mixed.sh`](../../../mixed.sh) (v2), 25 Nov 2026 (UTC). It repeats the [9 Nov run](../mixed-5090-2026-11-09) on a second machine of the same build, and adds a burst phase: four long prompts arriving at the same moment, the way several agents can finish a tool call together.

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04 (vast.ai, machine m:155807, listed in South Korea, datacenter 406325), $0.545/hour
- Host: AMD Ryzen 7 7800X3D, 16 threads, B850 EAGLE ICE, the same build as the 9 Nov machine (m:149075). Its disk read 411 MB/s on the listing, against about 6,000 MB/s on the 9 Nov machine
- vLLM 0.27.1 (official image), model `Qwen/Qwen2.5-7B-Instruct`
- Load, 60 s per phase, every answer streamed, 256 tokens (`ignore_eos`):
  - short only: 16 requests at once, prompts of about 512 tokens
  - mixed: the same 16, plus one request with a prompt of about 12,000 tokens every 10 s (6 per phase)
  - burst: the same 16, plus four 12,000-token requests sent together every 10 s (24 per phase)
- Two server configs, restarted between them:
  1. default flags
  2. `--max-num-batched-tokens 16384` (the whole 12,000-token prompt fits in one step, so it is not split)
- New in v2: a "budget" line per config, computed by vLLM's own argument parser and `EngineArgs` for the same flags. Default: `max_num_batched_tokens 2048, chunked prefill on, max_num_seqs 256`. Config 2: `16384`, chunked prefill on. This is vLLM's own config code run on this machine, not the server's log line (the server logged the line only for config 2)
- Meter: `qvunex_single.py` md5 5b49a0fa; script `mixed.sh` md5 45143146

## Results: the burst phase

Short answers while four long prompts arrive together every 10 s, default vs 16,384 budget:

- Biggest pause inside an answer, p95: 0.185 s vs 1.329 s
- Short answers with a pause over 0.3 s: 0 of 173 vs 100 of 160
- Whole answer: p50 7.29 s / p95 7.70 s vs p50 7.56 s / p95 7.96 s
- Time per streamed token (median over answers, after the first token): 27.1 ms vs 27.3 ms, against 11.9 and 11.0 ms with no long prompts

Long prompts in the burst, first token: p50 2.99 s / max 4.22 s vs p50 3.27 s / max 4.40 s. Within each burst the first of the four got its first token after about 1.0 to 1.3 s and the others after about 2, 3 and 4 s, with either setting: the four prompts were read one after another, about a second each.

Cost per 1M output tokens: $0.192 vs $0.202 (short only: $0.127 both).

0 errors in every phase. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens in all six phases. Every request returned 256 output tokens.

## Results: the mixed phase again

One long prompt every 10 s, default vs 16,384 budget: biggest pause p95 0.180 s vs 0.968 s; answers with a pause over 0.3 s 0 of 240 vs 113 of 240; long prompt first token about 1.0 s both ways; $0.146 vs $0.148 per 1M output tokens.

## Same build, two machines

The 9 Nov run (m:149075) and this one (m:155807), same flags and load:

- Short only, biggest pause p95: 0.134 vs 0.133 s (default), 0.528 vs 0.526 s (16,384)
- Mixed, biggest pause p95: 0.180 vs 0.180 s (default), 0.973 vs 0.968 s (16,384)
- Mixed, long prompt first token: 0.992-1.001 vs 0.99-1.00 s (default)
- Short only, cost per 1M output tokens: $0.122 at $0.522/hour vs $0.127 at $0.545/hour, the same cost once the price difference is taken out

Start-up was 131.8 s against 142.9 s on the 9 Nov machine, despite the slower disk; the model download (201 s) is not counted in start-up.

## What it shows

With default flags, a burst of long prompts did not freeze the short answers, but it slowed them down. Every streamed token took about 2.3 times as long, and whole answers took about twice as long as with no long prompts. Chunked prefill spreads the cost of reading long prompts over everyone's tokens instead of stopping everyone at once.

With the whole prompt read in one step, the same burst produced pauses of about 1.3 s inside most short answers, the long requests got their first token slightly later, and cost per token was slightly higher. The 16,384 budget was ahead on only two numbers in the whole run: whole-answer p95 with no long prompts (3.74 vs 3.90 s) and whole-answer p50 in the mixed phase (3.94 vs 4.03 s). The default had a much faster first token for short answers in every phase (p50 0.33 vs 0.57 to 0.59 s).

Four long prompts arriving together queue behind each other. The last of the four waited about 4 s for its first token, with either setting.

## Limits

- One run per phase. Four long prompts per burst, at a fixed 10 s interval, all the same length. Real agent traffic varies in size and timing.
- One card, one model, synthetic prompts (unique prefix, then a repeated word).
- The budget line is computed with vLLM's own config code on this machine, in a separate process before the server starts; it is not read from the running server.
- No config with chunked prefill turned off (in vLLM 0.27.1 that logs that it "may cause the engine to crash" for this model).
- vLLM logged a one-time Triton kernel compile at the first requests of each server (15:43:23 and 15:47:28).
- Costs assume the GPU is busy the whole window.

## Files

- the `mixed-files` archive (tar.gz; open with `tar xzf`): both server logs, /metrics before and after each phase, per-request timings (`requests_*.jsonl`), meter events, reconcile output, the report, and the exact `mixed.sh`, `budget.py` and meter that ran
