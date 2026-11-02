# Served-runs corpus: cost per token at a measured answer time

One row per load level of a served run (`planb.sh`): a real `vllm serve`, real
requests in flight, latency measured per request, cost from the GPU's hourly
price and the measured window. Every row's request and token counts were
checked against vLLM's own /metrics counters (`meter_matches_server`).

`served-runs.csv` has 9 rows from 2 runs so far: RTX 3090 (27 Oct 2026) and
A100 80GB PCIe (29 Oct 2026), same model, same vLLM, same load. The raw files
for every row are in the folder named in `source`.

## Which GPU is cheapest at your answer time?

```bash
python3 cheapest.py --p95 8
python3 cheapest.py --p50 4 --price "A100=1.40" --price "3090=0.22"
```

It prints, per GPU, the cheapest measured setup that meets your target, and
the GPUs that never met it. `--price` re-prices a GPU at your own hourly rate.
Standard library only.

At a p95 of 8 s: A100 at 64 at once, $0.106 per 1M output tokens; RTX
3090 at 4 at once, $0.490. With no latency limit, both reach $0.082.

## What it does not do

- Only measured points. Nothing is interpolated between load levels, and
  nothing carries over to another GPU, model, engine version or prompt length.
- Costs assume the GPU is busy the whole window. Idle hours in a real day raise
  them; only real traffic shows by how much.
- One run per level, synthetic prompts (unique prefix, `ignore_eos`).

## Columns

`run_date`, `gpu`, `gpu_mem_mib`, `driver`, `host` (provider and machine),
`usd_per_hour`, `engine`, `engine_version`, `model`, `extra_flags` (empty =
defaults), `prompt_tokens_approx`, `output_tokens`, `ignore_eos`,
`concurrency` (requests in flight), `window_s` (paid window, until the last
in-flight request finished), `requests_done`, `errors`, `output_tok_per_s`,
`latency_p50_s`, `latency_p95_s`, `usd_per_request`,
`usd_per_1m_output_tokens`, `meter_matches_server`, `startup_s` (launch to
/health, model on disk), `source`.

New rows come from new `planb.sh` runs, including client mini-reports when the
client agrees to publish.
