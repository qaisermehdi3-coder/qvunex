# Runbook — one card on vast.ai

Written before renting anything. The point is that no decision gets made while a
GPU is billing you.

Order: 3090 first. Two of the four cards are 24GB, and the 7B fp16 batch-128
question is unresolved. If 24GB genuinely cannot do it, that changes what is
deliverable on the 4090 too, and Seth asked to be told before the rental moves on.

---

## 1. Choosing the instance

Filter for a single RTX 3090. Then check these four before clicking rent — three
of them have cost real money before.

| check | why | minimum |
|---|---|---|
| **Disk space** | Four model downloads live on disk at once: 7B fp16 ~15GB, 7B AWQ ~5.5GB, 1.5B fp16 ~3.1GB, 1.5B AWQ ~1.2GB, plus the vLLM image ~10GB. vast.ai's default allocation is often smaller than that and the sweep dies mid-way with no rows. | **60 GB** |
| **Download bandwidth** | 25GB of weights arrive before the first timing. A slow host bills you for the download. | 300 Mbps+ |
| **Reliability score** | An instance that dies mid-sweep loses the session, and both model sizes must run in one session. | 99%+ |
| **Price / hour** | Whole sweep is roughly 2–4 hours. Note the exact figure for the conditions note. | — |

Do **not** pick a machine sharing the GPU with other tenants. Host CPU
contention is exactly the variable that moves eager timings 38–51%.

## 2. Image

```
vllm/vllm-openai:v0.27.1
```

Pinned by image, not by pip. The image's own entrypoint starts an API server, so
if vast.ai gives an entrypoint field, clear it — you want a shell, not a server.

## 3. Setup, once connected

```bash
# sweep.py shells out to `python`; the image only ships python3
ln -sf $(command -v python3) /usr/local/bin/python

apt-get update -qq && apt-get install -y -qq git

git clone --depth 1 https://github.com/qaisermehdi3-coder/qvunex.git /qvunex

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0

nvidia-smi                    # record card name and driver for the conditions note
df -h /workspace              # confirm the disk you actually got
lscpu | head -20              # record host CPU
```

Check `df -h` before starting. If it is under 60GB, destroy the instance and pick
another. That decision costs a few cents now and a whole sweep later.

## 4. The run

Both sizes, one session, small first so a disk or setup problem surfaces cheaply.

```bash
mkdir -p /workspace/out

# 1.5B
python3 /qvunex/benchmarks/sweep.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --compare-quantization awq \
  --batches 1,8,32,128 --modes eager,graphs \
  --max-model-len 1024 --repeats 3 \
  --out /workspace/out/3090-1p5b.csv

# 7B — the one that failed on the L4
python3 /qvunex/benchmarks/sweep.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --compare-model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --compare-quantization awq \
  --batches 1,8,32,128 --modes eager,graphs \
  --max-model-len 1024 --repeats 3 \
  --out /workspace/out/3090-7b.csv
```

`--repeats 3` is what turns each config into a band instead of a number. That is
the thing being sold; do not drop it to save time.

Run inside `tmux` or `screen` so a dropped connection does not kill the sweep.

## 5. If the 7B run fails

Expected at batch 32 or 128 on fp16. Capture, in this order:

1. `nvidia-smi` output at the moment of failure if you can get it
2. Whether the process **raised** or was **killed**
   - raised → there is an `error` row in the CSV, copy it verbatim
   - killed → no row exists, and that absence is the evidence. `dmesg | tail -40`
     will usually show an OOM kill and say whether it was host RAM or the card
3. `free -g` — if host RAM is the limit, it is the container, not the card

Then stop and email Seth before touching the next card. He asked for exactly
this, and said a clear partial failure is more useful than nothing.

## 6. Finishing

```bash
# pull the CSVs off before anything else
cat /workspace/out/3090-1p5b.csv
cat /workspace/out/3090-7b.csv
```

Copy them out, then **destroy** the instance — do not stop it. A stopped
instance keeps charging storage, and at zero balance it is not destroyed
automatically.

Fill in the conditions note from `CONDITIONS_TEMPLATE.md` while the run is still
fresh. Section 5 needs the cap you used *and* the cap this card could have
reached; section 6 needs any failure named as container memory or card VRAM.
