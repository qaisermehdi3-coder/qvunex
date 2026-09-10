# Conditions note — <CARD NAME>

One of these per card. Filled in from the CSV where possible; anything the sweep
cannot observe is stated here by hand or marked MISSING. A number without this
note is not comparable to any other number.

Fill the blanks. Do not delete a field you could not answer — write MISSING and
say why. A gap that is visible is a caveat; a gap that is hidden is a lie.

---

## 1. Hardware

| field | value |
|---|---|
| GPU name (from `nvidia-smi`) | |
| GPU count | 1 |
| VRAM | |
| Driver version | |
| Host CPU model | |
| Host CPU cores visible | |
| Provider and instance type | vast.ai, |
| Instance ID / host ID | |
| Price paid per GPU-hour | |

Host CPU is recorded because eager-mode timings track it and CUDA-graph timings
do not. Two sweeps of the same card on different hosts moved 38–51% on eager
rows and stayed within 1.3% on fp16 graph rows.

## 2. Software

| field | value |
|---|---|
| vLLM version | 0.27.1 |
| How it was pinned | container image `vllm/vllm-openai:v0.27.1` |
| torch version | |
| CUDA version | |
| Python version | |

Pinned by image rather than by pip: pinning vLLM while letting its dependencies
float produced a different stack on every rebuild.

## 3. Models

| field | value |
|---|---|
| fp16 model | |
| 4-bit model | |
| Quantization method | AWQ |
| dtype | |

## 4. Workload

| field | value |
|---|---|
| Batch sizes | 1, 8, 32, 128 |
| Prompt tokens per request | |
| Output tokens per request | 128 |
| `ignore_eos` | True — every request generates exactly `max_tokens` |
| Prompt prefixes | unique per request |
| Prefix caching | disabled |
| Seed | 0 |
| Warmup runs before timing | 1 |
| Repeats per config | |

## 5. Memory and context cap

| field | value |
|---|---|
| `gpu_memory_utilization` | 0.90 |
| `max_model_len` **actually used** | |
| `max_model_len` this card could have reached | |
| Same cap across all four cards? | |

Seth asked for both numbers explicitly. If a card needs a lower cap than the
others to fit batch 128 at 7B fp16, state the natural limit and the cap used,
and say which configs the difference affects.

## 6. Failures

Every config that did not produce a timing, and what happened.

| config | what happened | container memory or card VRAM? | evidence |
|---|---|---|---|
| | | | |

A config that fails is a row in the CSV with `status` set and `error` filled in,
not a missing row. If a run was killed rather than raising, say so — a kill
leaves no error row, and its absence is itself the evidence.

## 7. Session

| field | value |
|---|---|
| Date of session (UTC) | |
| Both model sizes on the same host, same session? | |
| `run_id` in the CSV | |
| Total wall time | |
| Total rental cost | |

Both model sizes must run on one host in one session. A host swap moved eager
timings 42% in earlier measurements, which would sit directly on top of the
model-size difference being measured.

## 8. What this note does not cover

State anything a reader might assume and be wrong about. At minimum:

- Single model family (Qwen2.5). Ratios may not transfer to other architectures.
- Short outputs (128 tokens). Do not extrapolate to long-context work.
- Cost figures are a rate times a time, not a measurement. Substitute your own rate.
- <anything else that surprised you during this run>
