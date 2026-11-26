# RTX 5090: two long prompts at once on a hybrid model (vllm#54919)

Three runs of [`longpair.sh`](../../../longpair.sh), 25 Nov 2026 (UTC), to check [vllm-project/vllm#54919](https://github.com/vllm-project/vllm/issues/54919) on one GPU. In that issue, an answer that is already streaming almost stops while a second long prompt is read. One comment there reproduced it with this model on an RTX 4080 SUPER.

## Setup

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04 (vast.ai, South Korea, datacenter 406325), $0.505/hour. Run 1 on machine m:100676; runs 2 and 3 together on a second rental from the same listing, same build
- Host: AMD Ryzen 7 7800X3D, 16 threads
- vLLM 0.27.1 (official image), model `cyankiwi/Qwen3.8-27B-AWQ-INT4` (hybrid: linear-attention and full-attention layers), no MTP
- Base flags, from the issue comment: `--max-model-len 200000 --max-num-seqs 3 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --enable-prefix-caching`
- Load: a control pair (2 prompts of about 500 tokens), then the long pair: A (about 60,000 tokens) and B (about 75,000 tokens) sent together. Every answer 1,024 tokens (`ignore_eos`, temperature 0), streamed through the completions API
- The three runs change one thing each:
  1. base flags
  2. base flags without `--enable-prefix-caching`
  3. base flags with `--max-num-batched-tokens 1024`
- The server log confirms the budget in each run (`Chunked prefill is enabled with max_num_batched_tokens=4096`, `4096`, `1024`)
- Meter: `qvunex_single.py` md5 5b49a0fa; script `longpair.sh` md5 9591eb5a

## Results

A's answer while B's prompt was being read (run 1 / run 2 / run 3):

- A's decode speed: 1.08 / 0.73 / 3.23 tokens per second (over 24.2 / 24.6 / 23.8 s)
- A's longest pause between two tokens: 1.30 / 1.80 / 0.42 s
- A's speed after B's first token: 60.5 / 60.7 / 60.4 tokens per second

First token of the long prompts:

- A: 20.6 / 20.5 / 24.6 s
- B: 44.8 / 45.1 / 48.4 s
- Whole pair: 61.6 / 61.9 / 65.2 s

Control pair: about 72 tokens per second each in all three runs. 0 errors. Meter vs vLLM's own counters: +0 on requests, prompt tokens and output tokens in every phase.

## What it shows

The slowdown from the issue happens on this card too: A dropped from about 60 to about 1 token per second for the 24 s it took to read B.

Turning prefix caching off did not remove it (run 2 was slightly worse; the [26 Nov sweep](../lpsweep-5090-2026-11-26) found why, see the correction below).

The numbers fit a simple count. While B is read, each engine step holds one chunk of B's prompt, up to the token budget, plus one token for A. So A gets about one token per step:

- Budget 4096: B's ~75,000 tokens take ~18 steps in ~24 s, about 1.3 s per step, so A gets about 1 token per second (this fits run 2; see the correction below for run 1)
- Budget 1024: ~73 steps in ~24 s, about 0.33 s per step, so A gets about 3 tokens per second

Run 3 was chosen from this count after runs 1 and 2: it predicted roughly 3 to 4 tokens per second for A with a 1024 budget. The run gave 3.23.

The smaller budget cut A's longest pause from 1.30 to 0.42 s and tripled its speed while B was read, at the cost of about 4 s more before each long prompt's first token (+19% for A, +8% for B).

## Correction (27 Nov 2026)

The count above fits run 2 (prefix caching off) but not run 1. With prefix caching on, this hybrid model runs Mamba cache "align" mode with an attention block of 1,568 tokens, and vLLM 0.27.1's scheduler cuts each prefill chunk to end on a multiple of 1,568. So in run 1 a 4096 budget most likely gave 3,136-token chunks: about 24 steps of about 1.0 s, which fits A's 1.08 tokens per second, and is why run 1 was faster than run 2 (0.73). The [26 Nov sweep](../lpsweep-5090-2026-11-26) tested this on six settings: one token per step held in all six, and with prefix caching on the chunk sizes matched the 1,568 rule, apart from a few chunks at 4096 that split in two for a reason not yet known.

## Limits

- One run per setting, one pair of prompts, synthetic text (a repeated word). Completions API with `ignore_eos`; the issue's repro used chat completions without it
- Seconds here, not the minutes reported in the issue on 2-node DGX Spark. Whether the same count explains that case is not known
- Run 1 and runs 2-3 were on two rentals of the same build, not the same machine
- Not tested: MTP, budgets other than 4096 and 1024, other models, SGLang
- Seen but not explained: during A's prefill the server showed B as waiting in runs 1 and 2 but as running in run 3, yet B's first token still came about 24 s after A's in all three. Peak KV cache use was 96% in run 1 against 57% and 58% in runs 2 and 3

## Files

`longpair-runs.tgz` (a phone download may name it `longpair-runs.tgz.gz`; it is the same normal tar.gz) holds one folder per run (`run1-4096-cache-on`, `run2-4096-cache-off`, `run3-1024-cache-on`), each with the report, per-request token times, /metrics polls every 3 s, the meter's events and reconcile, and the server log.
