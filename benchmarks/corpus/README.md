# Served-runs corpus: cost per token at a measured answer time

One row per load level of a served run (`planb.sh`): a real `vllm serve`, real
requests in flight, latency measured per request, cost from the GPU's hourly
price and the measured window. Every row's request and token counts were
checked against vLLM's own /metrics counters (`meter_matches_server`).

`served-runs.csv` has 23 rows from 5 runs so far: RTX 3090 (27 Oct 2026),
A100 80GB PCIe (29 Oct 2026), RTX 5090 and H100 80GB SXM (both 7 Nov 2026),
and the RTX 5090 again on a second host (8 Nov 2026), same model, same vLLM,
same load. The raw files for every row are in the folder named in `source`.

## Which GPU is cheapest at your answer time?

```bash
python3 cheapest.py --p95 8
python3 cheapest.py --p50 4 --price "A100=1.40" --price "3090=0.22"
```

It prints, per GPU, the cheapest measured setup that meets your target, and
the GPUs that never met it. `--price` re-prices a GPU at your own hourly rate.
A GPU measured on several hosts is compared at one price, the lowest measured
for it unless you pass `--price`; a row measured at another price says so.
Standard library only.

At a p95 of 8 s: RTX 5090 at 64 at once, $0.048 per 1M output tokens; H100 at
256 at once, $0.099; A100 at 64 at once, $0.106; RTX 3090 at 4 at once, $0.490.
At a p95 of 3 s: H100 at 64 at once, $0.162; RTX 5090 at 4 at once, $0.382.
Under 2 s, only the H100 was fast enough. With no latency limit: RTX 5090 at
256 at once, $0.038 (p95 24.3 s); the 3090 and A100 both reach $0.082.

The two RTX 5090 hosts served within about 4% of each other at every load,
but start-up took 142 s on one and 249 s on the other
([details](../data/planb-5090-2026-11-08)).

## What it does not do

- Only measured points. Nothing is interpolated between load levels, and
  nothing carries over to another GPU, model, engine version or prompt length.
- Costs assume the GPU is busy the whole window. Idle hours in a real day raise
  them; only real traffic shows by how much.
- One run per level, synthetic prompts (unique prefix, `ignore_eos`).

## Columns

`run_date`, `gpu` ("4x NAME" on a machine with four of the same GPU),
`gpu_mem_mib` (per GPU), `driver`, `host` (provider and machine),
`usd_per_hour`, `engine`, `engine_version`, `model`, `extra_flags` (empty =
defaults), `prompt_tokens_approx`, `output_tokens`, `ignore_eos`,
`concurrency` (requests in flight), `window_s` (paid window, until the last
in-flight request finished), `requests_done`, `errors`, `output_tok_per_s`,
`latency_p50_s`, `latency_p95_s`, `usd_per_request`,
`usd_per_1m_output_tokens`, `meter_matches_server`, `startup_s` (launch to
/health, model on disk), `source`.

New rows come from new `planb.sh` runs, including client mini-reports when the
client agrees to publish.

Since 4 Nov 2026 `planb.sh` writes them for you as `corpus-rows.csv` (fill in `host` and `source`).

## `offline-batch.csv`: the earlier sweeps, a different method

240 rows from the August to October sweeps (`sweep.py`, `run3090.sh`): Tesla T4, L4 on two hosts, RTX 3090 and L40S, Qwen2.5 1.5B and 7B with their AWQ builds, CUDA graphs on and off, batch 1, 8, 32 and 128, 128 output tokens, short prompts (about 74 tokens). Every row points to its original file in `source`.

**Do not compare these rows with `served-runs.csv`.** These are offline batches: one wave of `batch` requests started together inside vLLM, no HTTP server, no requests arriving over time, no per-request latency. `seconds_total` is the time for the whole wave. Use them to compare cards and settings with each other.

`usd_per_1m_output_tokens` is filled only where the hourly price was recorded (T4 at $0.35, RTX 3090 at $0.414); for L4 and L40S it is empty rather than guessed. Read the caveats in `../data/README.md` first: eager timings moved up to 50% between hosts, CUDA-graph fp16 timings stayed within 1.3%.

Columns: `run_date`, `gpu`, `host`, `usd_per_hour`, `engine`, `engine_version`, `model`, `quantization`, `cuda_graphs`, `batch`, `output_tokens_each`, `repeat`, `seconds_total`, `output_tok_per_s`, `usd_per_1m_output_tokens`, `board_watts` (T4 only), `source`.

## Add your GPU's numbers

One command on a Linux machine with an NVIDIA GPU, Python with `vllm`, `openai` and `huggingface_hub` installed, and about 15 GB free for the model (or a rented GPU with the official vLLM image):

```bash
curl -sL https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/planb.sh | bash -s -- YOUR_PRICE_PER_HOUR
```

It serves Qwen2.5-7B-Instruct with your vLLM on port 8000 (set `QVUNEX_PORT` to change it; it stops if the port is already in use), sends 1, 4, 16 and 64 requests at once for 60 s each, checks every count against vLLM's own /metrics, and writes `corpus-rows.csv` to /workspace/planb if that folder exists, otherwise ~/qvunex-planb (or to `QVUNEX_OUT`). Inside the official vLLM image it also copies the report, `corpus-rows.csv` and the raw files to /vllm-workspace. A model already in the machine's Hugging Face cache (`HF_HOME`) is reused, and a local model folder can be given instead of a name. Your own model and vLLM flags go after the price, for example `bash -s -- 1.20 YOUR_MODEL 512 256 "1 4 16 64" 60 "--tensor-parallel-size 4"` on a machine with four GPUs. It uses the whole GPU for about 10 minutes, so don't run it on a GPU that is serving users. On your own machine it stops only the server it started.

Then open an [Add GPU numbers](https://github.com/qaisermehdi3-coder/qvunex/issues/new?template=add-gpu-numbers.yml) issue and paste the file. Rows are checked and added with your name as the source.
