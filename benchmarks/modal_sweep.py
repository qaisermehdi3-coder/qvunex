"""Rehearsal for the SlickToken job: 7B alongside 1.5B, one host, one session.

Run this before spending a client's money. It exercises the exact thing most
likely to break on the paid cards -- a 7B model at fp16 on 24GB with batch 128 --
on an L4, which has the same 24GB as the RTX 3090 and RTX 4090.

Both model sizes run inside ONE container invocation on purpose. A host swap
moved eager timings 42.1% in earlier measurements, so if the 1.5B and the 7B ran
on different machines, host variance would sit directly on top of the model-size
difference we are trying to measure. Same host, same session, both sizes.

    !pip install -q modal
    !modal setup
    !modal run --detach qvunex_7b_rehearsal.py

Then fetch the results:

    !modal volume get qvunex-sweeps rehearsal --force
"""

import modal

app = modal.App("qvunex-7b-rehearsal")

# The official vLLM image, pinned. Building our own from pip is what caused
# three days of container failures before: pinning vllm while letting its
# dependencies float meant a different stack every rebuild.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.27.1", add_python=None)
    .entrypoint([])                       # the image's entrypoint starts a server
    .apt_install("git")
    .run_commands(
        # sweep.py shells out to `python`; the image only has python3.
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "git clone --depth 1 https://github.com/qaisermehdi3-coder/qvunex.git /qvunex",
    )
)

out_vol = modal.Volume.from_name("qvunex-sweeps", create_if_missing=True)
# Weights cached on a volume so 32 fresh subprocesses do not re-download a 15GB
# model 32 times. Each config still reloads from local disk, which is the point:
# a cold load per config is what keeps configs independent.
hf_vol = modal.Volume.from_name("qvunex-hf-cache", create_if_missing=True)

SIZES = [
    # tag,      fp16 model,                        awq model
    ("1p5b", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct-AWQ"),
    ("7b",   "Qwen/Qwen2.5-7B-Instruct",   "Qwen/Qwen2.5-7B-Instruct-AWQ"),
]

# 7B at fp16 is ~15.2GB of weights. On a 24GB card at 0.90 utilisation that
# leaves roughly 5GB for KV, and KV is ~56KB/token on this architecture, so about
# 90,000 tokens of budget. Batch 128 needs ~27,000. It fits. The cap is stated
# explicitly and used on every card so the numbers stay comparable.
MAX_MODEL_LEN = 1024


# memory and cpu are explicit because the first rehearsal died here. A 7B model
# at fp16 streams ~15GB of weights through host RAM on every load, and this
# script loads once per config on purpose. Batch 1 and 8 survived on the default
# allocation; batch 32, which adds CUDA-graph capture on top, did not -- and the
# container was killed mid-traceback rather than raising, which is what a memory
# kill looks like from the outside.
@app.function(gpu="L4", image=image, timeout=86400,
              memory=32768, cpu=8.0,
              volumes={"/out": out_vol, "/hf": hf_vol})
def run_both_sizes(repeats: int = 1, only: str = ""):
    import os
    import subprocess

    env = dict(os.environ)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["HF_HOME"] = "/hf"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    os.makedirs("/out/rehearsal", exist_ok=True)

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    wanted = [s for s in SIZES if not only or s[0] == only]
    print("running sizes:", [s[0] for s in wanted], flush=True)

    results = {}
    for tag, fp16_model, awq_model in wanted:
        out_csv = f"/out/rehearsal/l4-{tag}.csv"
        cmd = [
            "python3", "/qvunex/benchmarks/sweep.py",
            "--model", fp16_model,
            "--compare-model", awq_model,
            "--compare-quantization", "awq",
            "--batches", "1,8,32,128",
            "--modes", "eager,graphs",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--repeats", str(repeats),
            "--out", out_csv,
        ]
        print("\n" + "=" * 70)
        print("  " + " ".join(cmd))
        print("=" * 70, flush=True)

        proc = subprocess.run(cmd, env=env)
        results[tag] = proc.returncode
        out_vol.commit()          # commit after each size, so a later failure
                                  # does not lose the earlier results

        if os.path.exists(out_csv):
            with open(out_csv) as fh:
                rows = fh.read().strip().splitlines()
            print(f"\n{tag}: {len(rows) - 1} rows written to {out_csv}")
            ok = sum(1 for r in rows[1:] if ",ok," in r or r.rstrip().endswith(",ok,"))
            print(f"{tag}: return code {proc.returncode}")
        else:
            print(f"\n{tag}: NO CSV WRITTEN -- this is the failure to investigate")

    out_vol.commit()
    return results


@app.local_entrypoint()
def main(repeats: int = 1, only: str = ""):
    codes = run_both_sizes.remote(repeats=repeats, only=only)
    print("\nreturn codes:", codes)
    print("fetch with:  modal volume get qvunex-sweeps rehearsal --force")
