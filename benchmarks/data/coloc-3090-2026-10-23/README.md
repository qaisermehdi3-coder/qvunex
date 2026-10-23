# Two vLLM servers on one GPU (RTX 3090, vLLM 0.27.1)

Measured 23 Oct 2026 for [vllm-project/vllm#60599](https://github.com/vllm-project/vllm/issues/60599):
can a small embedding model share one GPU with a chat model, each in its own `vllm serve`?

## Setup

- GPU: NVIDIA GeForce RTX 3090, 24576 MiB, driver 580.173.02 (vast.ai, host 443829, California)
- vLLM 0.27.1 (official image), 64 CPU threads visible
- Embedding: `BAAI/bge-small-en-v1.5`, `--max-model-len 512` (vLLM resolved it to `--runner pooling`)
- Chat: `Qwen/Qwen2.5-1.5B-Instruct`, `--max-model-len 2048`, CUDA graphs on (default)
- Load: embeddings 8 requests at once, 32 texts each; chat 16 requests at once, 128 output tokens each (`ignore_eos`)
- Script: [`coloc.sh`](../../../coloc.sh) at commit 41db042 (md5 7baca0c5)

## What happened

1. Embedding server alone at 0.10: ready in 66 s, 556 MiB on the GPU, 1,733 texts/s.
2. Chat server started second at 0.70: ready in 82 s. Both on one GPU: 18,317 MiB in use (embedding 578 MiB, chat 17,710 MiB).
3. Throughput, each busy alone vs both busy at the same time (30 s each, 0 errors):
   - embeddings: 1,760.6 -> 1,609.1 texts/s (-8.6%)
   - chat: 2,606.7 -> 2,298.3 output tokens/s (-11.8%)
4. Chat started first at 0.90, then embedding at 0.10: the embedding server refused to start.
   vLLM's own message (captured in the re-run, `d-rerun.txt`):
   `Free memory on device cuda:0 (0.95/23.8 GiB) on startup is less than desired GPU memory utilization (0.1, 2.38 GiB).`
5. Both started at the same moment (0.70 + 0.10): both ready (38 s and 49 s).
6. Embedding alone at 0.05 and at 0.03 (with `--max-num-seqs 32`): 458 and 454 MiB, 1,635 and 1,763 texts/s.
7. Re-run: chat at 0.9 first, then embedding at **0.03**: started, `/health` 200, chat 23,110 MiB + embedding 552 MiB.

## What it means (for these two models, this card, this version)

- Separate servers with split `--gpu-memory-utilization` work.
- The embedding model's log has no KV cache allocation line (the chat model's does), and its memory stayed at about
  450-580 MiB whether its fraction was 0.10, 0.05 or 0.03. So for it the fraction only has to be large enough to pass
  vLLM's start-up check; it does not reserve that memory.
- That start-up check is the real limit: a server will not start if free memory is below its fraction of the whole GPU.
  If another server already took nearly everything, give the late one a fraction just above its real need.
- The chat server used more than its fraction: 17,710 MiB at 0.70 (17,203 MiB requested) and 23,110 MiB at 0.9.
  vLLM's own log at 0.70 says so too ("--kv-cache-memory ... (11.9 GiB) to fit into requested memory ... Current kv cache
  memory in use is 12.53 GiB"). Leave headroom.

## Limits

One card, one vLLM version (0.27.1), one pair of small models, 30-second load windows, one run of each test.
Not tested: newer vLLM, larger models, `--enforce-eager`, MPS/MIG, Kubernetes.

## Files

- `results-coloc.txt` - the script's output, unedited
- `d-rerun.txt` - test 4 repeated by hand to capture the error line (transcribed from a screenshot)
