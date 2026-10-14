# Self-hosted vLLM: what the client is told vs what the server counted

One live run, raw files unedited. Colab T4, 14 October 2026, 17:40 UTC.

## Conditions

- GPU: Tesla T4, driver 580.82.07 (Colab free tier)
- vLLM 0.27.1 (pip), torch 2.13.0+cu130, Python 3.13.15
- Model: Qwen/Qwen2.5-0.5B-Instruct, `--dtype half`, all other server flags default
  (so prefix caching on, `--enable-prompt-tokens-details` off; see `server.log` line 13)
- Client: `openai` Python SDK through `qvunex_single.py` (sha256 in `conditions.json`)
- The same request three times, one after another: "Name three rivers in Europe.",
  `max_tokens` 16, `temperature` 0, `ignore_eos` true
  - A: plain call
  - B: stream, no `stream_options`
  - C: stream with `stream_options={"include_usage": true}`
- `/metrics` saved right before A (`before.txt`) and after C (`after.txt`), from a
  fresh server, so `before.txt` holds zeros

## What each surface saw

Client, straight from the response objects (`conditions.json`, `calls_in_order`):

- A: 35 prompt, 16 output, no `prompt_tokens_details`
- B: no usage at all
- C: 35 prompt, 16 output, no `prompt_tokens_details`

Server, `after.txt` minus `before.txt`:

- `vllm:request_success_total` 3
- `vllm:request_prompt_tokens_sum` 105, `vllm:prompt_tokens_total` 105
- `vllm:request_generation_tokens_sum` 48, `vllm:generation_tokens_total` 48
- `vllm:prompt_tokens_cached_total` 64 (also `vllm:prefix_cache_hits_total` 64 of
  `vllm:prefix_cache_queries_total` 105; `server.log` reports a prefix cache hit rate of 61.0%)

## What follows

1. Totals reconcile. Server 3 / 105 / 48, client 2 known calls 70 / 32. The gap,
   35 / 16, is exactly one call's worth: the stream that got no usage. (`reconcile.txt`)
2. Absent is not zero. B came back with no usage object; it is recorded as
   `usage_missing`, never as 0 tokens (`events.jsonl`, line 3).
3. Attribution here is only by elimination. With exactly one call missing, the
   gap must be that call's. With two or more missing, the server's sums cannot say
   which call used how much.
4. The cache split is not on the client surface. The server read 64 of 105 prompt
   tokens from its prefix cache. The client was told nothing about caching on any
   call, because vLLM sends `prompt_tokens_details` only with
   `--enable-prompt-tokens-details`. The counts the client did get (35 prompt) include
   cached tokens, so they are right; only the split is unknown.
5. Point 4 found a bug in `qvunex_single.py` itself: the meter that made
   `events.jsonl` recorded `cache_read: 0` for A and C, reading the absent field as
   zero. Fixed the same day: such calls are now marked as cache split unknown, and
   `--reconcile` prints the server's cached count on its own line. `events.jsonl`
   is left as the old meter wrote it.

## Files

| file | what it is |
|---|---|
| `before.txt`, `after.txt` | vLLM `/metrics`, saved with `curl` |
| `events.jsonl` | the meter's record of the three calls, as written live |
| `conditions.json` | versions, GPU, the request, and each call's usage as the client saw it |
| `reconcile.txt` | `qvunex_single.py --reconcile before.txt after.txt` output, live |
| `server.log` | the vLLM server log of the run |

Limits: one run, one small model, one card, three calls. It shows what the two
surfaces report, not how fast or how costly anything is.
