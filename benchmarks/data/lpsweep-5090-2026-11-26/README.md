# RTX 5090: token budget sweep, two long prompts at once (vllm#54919)

One run of [`lpsweep.sh`](../../../lpsweep.sh), 26 Nov 2026 (UTC). It runs [`longpair.sh`](../../../longpair.sh) six times on the same GPU, changing the token budget (`--max-num-batched-tokens`) and prefix caching. It follows the [three earlier runs](../longpair-5090-2026-11-25) for [vllm-project/vllm#54919](https://github.com/vllm-project/vllm/issues/54919).

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 610.57.04 (vast.ai, machine m:141416, Hungary), $0.632/hour
- Host: AMD Ryzen 7 9700X, 16 threads, PCIe 4.0 x16. A different machine, driver and CPU from the earlier runs
- vLLM 0.27.1 (official image), model `cyankiwi/Qwen3.8-27B-AWQ-INT4`, no MTP
- Flags in every run: `--max-model-len 200000 --max-num-seqs 3 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8`, plus the budget, plus `--enable-prefix-caching` for the two "on" settings
- Load as before: two prompts sent together, about 60,000 and 75,000 tokens, every answer 1,024 tokens, streamed
- Scripts: `lpsweep.sh` md5 2e246e7e, `longpair.sh` md5 9591eb5a, meter `qvunex_single.py` md5 5b49a0fa

## Results

Both prompts are sent at the same moment, and vLLM picks which one to read first. In three settings it read the 75k prompt first, in the other three the 60k one. "Waiting answer" is the one that got its first token first and then streamed while the other prompt was read.

- off:512: waiting answer 4.67 tokens/s over 31.7 s, longest pause 0.29 s; first tokens 23.6 s and 55.3 s
- off:1024: 2.55 tokens/s over 29.0 s, longest pause 0.56 s; first tokens 22.1 s and 51.0 s
- off:2048: 1.39 tokens/s over 20.9 s, longest pause 0.92 s; first tokens 28.5 s and 49.4 s (75k prompt read first)
- off:4096: 0.67 tokens/s over 27.0 s, longest pause 1.95 s; first tokens 22.4 s and 49.4 s
- on:3136: 1.85 tokens/s over 20.6 s, longest pause 0.69 s; first tokens 28.8 s and 49.4 s (75k prompt read first)
- on:4096: 1.20 tokens/s over 20.0 s, longest pause 1.33 s; first tokens 29.3 s and 49.2 s (75k prompt read first)

After both first tokens, every answer streamed at about 59 to 60 tokens per second. Every answer returned 1,024 tokens. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens in every phase.

## What it shows

**One token per step, in all six.** The waiting answer got one token for each chunk of the other prompt. Tokens it streamed in the window, against the other prompt's length divided by the chunk size:

- off:512: 148 tokens, 75,000 / 511 = 147 chunks
- off:1024: 74 tokens, 75,000 / 1,023 = 73 chunks
- off:2048: 29 tokens, 60,000 / 2,047 = 29 chunks
- off:4096: 18 tokens, 75,000 / 4,095 = 18 chunks
- on:3136: 38 tokens, 60,000 / 1,568 = 38 chunks (see below)
- on:4096: 24 tokens, 60,000 / 3,136 = 19 chunks (see below)

The chunk is the budget minus one: the waiting answer's token takes one place in the budget. The counts fit that.

**Steps get slower as the prompt gets longer.** Inside one window the time between the waiting answer's tokens kept rising: at 2048, from 0.56 s to 0.92 s.

**Prefix caching on changes the chunk size.** With prefix caching on, this hybrid model runs Mamba cache "align" mode, and the server log says "Setting attention block size to 1568 tokens". vLLM 0.27.1's scheduler (`_mamba_block_aligned_split`) cuts each prefill chunk so it ends on a multiple of 1,568. So:

- on:3136: the budget minus the waiting token is 3,135, which is cut down to 1,568. 38 chunks, 38 tokens. A 3136 budget gave half-size chunks, so the waiting answer ran faster than at 4096
- on:4096: 4,095 is cut down to 3,136. Most steps took about 0.9 to 1.3 s, as expected, but at least three times a chunk was split into two shorter steps (about 0.4 to 0.5 s, then 0.7 to 0.8 s), which gives 24 tokens instead of 19. Why those splits happened is not known

This most likely explains the earlier runs too: on 25 Nov, prefix caching on (1.08 tokens/s) was faster than off (0.73) at the same 4096 budget because its chunks were 3,136 tokens, not 4,095.

**The cost of a small budget was small here.** With prefix caching off, going from 4096 to 1024 made the waiting answer 3.8 times faster (2.55 vs 0.67 tokens/s) and cut its longest pause from 1.95 s to 0.56 s. The long prompt's first token came 1.6 s later (51.0 vs 49.4 s, +3%). At 512 it came 5.9 s later (+12%).

## Predictions made before the run

Written in the work log before the sweep started, from the one-token-per-step count:

- off:4096 about 0.7: got 0.67
- off:2048 about 1.5: got 1.39
- off:1024 about 3: got 2.55
- off:512 about 5 to 6, or less if per-step overhead grows: got 4.67
- on:4096 about 1.0 to 1.1: got 1.20
- on:3136 about the same as on:4096: wrong, got 1.85 vs 1.20. The prediction forgot that the waiting token takes one place in the budget, so 3,136 is cut to 1,568, not kept at 3,136

## Limits

- One run per setting, one pair of prompts, synthetic text (a repeated word). Completions API with `ignore_eos`
- One GPU, one model. Which prompt vLLM read first changed between settings, so three windows are 60k prompts and three are 75k
- This machine read prompts about 10% slower than the 25 Nov machine (4096, prefix caching off: 1.48 s per step vs 1.34 s); its driver and CPU differ
- The split chunks at on:4096 are not explained
- In two settings the script's own budget check (vLLM's EngineArgs) printed "unknown"; the server log line confirms the budget in five of six. At 2048, vLLM's default on this GPU, the server does not print that line, as seen in earlier runs

## Files

`sweep-files.tgz` (a phone download may name it `sweep-files.tgz.gz`; it is the same normal tar.gz) holds one folder per setting (`run1-off-512` to `run6-on-4096`), each with the report, every token's time, /metrics every 3 s, the meter's events and reconcile, and the server log, plus the combined `report.txt`.
