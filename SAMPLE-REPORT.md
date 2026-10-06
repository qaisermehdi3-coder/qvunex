# What a measured GPU cost report looks like

A sample, from runs already published in this repo. A report on your own
hardware has the same parts, measured on your machines at your prices.
Method: [METHOD.md](METHOD.md). Tool: [`qvunex_single.py`](qvunex_single.py).

## Sample: 1x H100 SXM 80GB, 28 September 2026

**Conditions.** Rented on vast.ai, 20 vCPUs, $2.689 per hour. vLLM 0.27.1
(official image), Qwen2.5-0.5B-Instruct, fp16, unique uncached prompts. Every
number below comes from vLLM's own `/metrics` counters, checked against what the
client recorded.

**1. One cold wake (scale to zero).** Clock started before the server launched.
The server answered after about 75 s of 77.3 s billed: about 97% of the wake
was boot. 8 requests served, $0.0072 per request. Meter and server counts were
equal.

**2. Speed and cost per token, by load.** 3 rounds each.

- one request at a time: per-request decode 648-649 tok/s, whole server 589-593
  tok/s, about $1.26 per 1M output tokens
- 8 requests at once: per-request decode 549-573 tok/s, whole server 3,194-3,348
  tok/s, about $0.22-0.23 per 1M output tokens

Priced from one request's speed, the same load looks about 5.7x dearer than it
was. Cost here is the hourly price over the whole window, idle inside the
window included, divided by what the server actually produced.

## The same checks on other servers and cards

- **T4, vLLM 0.27.1 (Colab):** boot about 165 of 171 s billed; whole server at
  8 at once 695-708 tok/s against 122-124 tok/s per request. Same 5.7x.
- **T4, SGLang 0.5.21 (Colab), 5 October 2026:** the health check said ready at
  387 s, but the first chat request then took about 176 s more. The first
  answer came about 563 s into a 640 s bill (88%). Warm requests: 0.08-0.22 s.
  Ready is not warm.
- **Paid, nothing served:** a window with only health checks and model-list
  calls is reported with its dollar amount and 0 requests, never as unknown.

## What a report on your hardware adds

- your GPUs, your prices, your vCPU and host details in a conditions note
- model sizes you choose (for example 1.5B, 7B, 70B-class), batch 1 to 128,
  3 repeats each, median reported
- cold start and the idle tail for serverless or scale-to-zero offers
- everything needed for anyone to repeat it

Contact: qvunexaudit@gmail.com
