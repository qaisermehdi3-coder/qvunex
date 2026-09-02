#!/usr/bin/env python3
"""
sweep.py - measure what one inference actually costs, under conditions you can publish.

The problem this exists for: two people run "the same model on the same GPU" and
report numbers that differ by 2x, 10x, sometimes more. Almost always the gap is a
flag nobody wrote down. This script sweeps the flags that move the number most,
and writes every condition into the CSV alongside the result, so the number can be
compared to somebody else's.

Usage:

    pip install vllm
    python sweep.py

    # your own model
    python sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --batches 1,8,32,128

    # compare two builds of the same model, e.g. fp16 against a 4-bit AWQ build
    python sweep.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --compare-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
        --compare-quantization awq

    # measure how repeatable your own numbers are (recommended, see below)
    python sweep.py --repeats 5

Output: a CSV you can open, read, and send to someone else without them having to
ask you six follow-up questions.

Design notes, because they matter for the numbers:

  * Each configuration runs in a FRESH SUBPROCESS. vLLM builds CUDA graphs at
    engine init, so you cannot honestly compare eager and graph mode inside one
    process. One process, one engine, one config.

  * Prefix caching is turned OFF, and every prompt in a batch gets a unique
    prefix anyway. With caching on and identical prompts, requests 2..N skip
    prefill entirely and your throughput number is measuring a cache, not a GPU.
    Both facts are recorded in the CSV so a reader can see it was handled.

  * ignore_eos=True with a fixed max_tokens, so every run generates exactly the
    same number of tokens. Otherwise you are comparing runs that did different
    amounts of work and calling it a speed difference.

  * The GPU name comes from nvidia-smi, not torch. Touching CUDA in the parent
    process before vLLM starts forces spawn mode and breaks the engine.

  * The CPU is recorded too, and that is not padding. Eager mode dispatches every
    operation individually from Python, so its wall time depends on the host CPU
    and how contended it is; CUDA graphs replay a pre-recorded schedule and barely
    touch the CPU. Measured on one L4: five runs in one container varied 0.23% with
    graphs and 5.9% eager, and the same eager config on a different host moved 42%
    while graphs moved 0.27%. If your eager numbers move and your graph numbers
    don't, look at this column before you blame the GPU.

  * --repeats runs the whole sweep N times and writes one row per repeat. A single
    measurement cannot tell you whether a difference is real. Use at least 3 before
    publishing any ratio.

  * A failing config records the error and the sweep continues. You get partial
    data instead of nothing.

MIT licensed. Part of qvunex: github.com/qaisermehdi3-coder/qvunex
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

FIELDS = [
    "run_id",
    "repeat",
    "timestamp_utc",
    "gpu_name",
    "gpu_count",
    "driver_version",
    "cpu_model",
    "cpu_cores",
    "vllm_version",
    "torch_version",
    "model",
    "quantization",
    "dtype",
    "enforce_eager",
    "prefix_caching",
    "gpu_memory_utilization",
    "max_model_len",
    "tensor_parallel_size",
    "batch",
    "prompt_tokens_each",
    "max_tokens",
    "ignore_eos",
    "seed",
    "warmup_runs",
    "seconds_total",
    "seconds_per_inference",
    "output_tokens_total",
    "output_tokens_per_second",
    "status",
    "error",
]


# ----------------------------------------------------------------------------
# environment probing (no torch, no CUDA - this runs in the parent)
# ----------------------------------------------------------------------------

def probe_gpus():
    """Read GPU name/count/driver from nvidia-smi.

    Deliberately not torch.cuda.get_device_name(): initialising CUDA in the
    parent process forces vLLM's engine child into spawn mode and it dies.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:
        return {"gpu_name": "unknown", "gpu_count": 0, "driver_version": "unknown"}

    rows = [r.strip() for r in out.splitlines() if r.strip()]
    if not rows:
        return {"gpu_name": "unknown", "gpu_count": 0, "driver_version": "unknown"}

    name, driver = (rows[0].split(",", 1) + [""])[:2]
    return {
        "gpu_name": name.strip(),
        "gpu_count": len(rows),
        "driver_version": driver.strip(),
    }


def probe_cpu():
    """Host CPU model and visible core count.

    Eager-mode timings track this closely; CUDA-graph timings do not. Recording
    it is what lets someone else tell those two cases apart in your data.
    """
    model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    if model == "unknown":
        try:
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True,
                                   timeout=10).stdout.strip() or "unknown"
        except Exception:
            pass

    model = re.sub(r"\s+", " ", model)
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 0
    return {"cpu_model": model, "cpu_cores": cores}


def make_prompts(count, target_words):
    """Distinct prompts of near-identical length.

    The unique id goes FIRST so that no two prompts share a prefix. Prefix
    caching is disabled as well, but a benchmark should not depend on a flag
    being honoured to stay correct.
    """
    body = " ".join(["token"] * max(1, target_words - 6))
    return [
        "Request %d unique %s . Continue writing: %s"
        % (i, uuid.uuid4().hex[:8], body)
        for i in range(count)
    ]


# ----------------------------------------------------------------------------
# the measured run - executed in its own process, one config only
# ----------------------------------------------------------------------------

def run_single(args):
    import torch  # noqa: F401  (imported here, in the child, never in the parent)
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        quantization=args.quantization or None,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        disable_log_stats=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,      # every request generates exactly max_tokens
        seed=args.seed,
    )

    prompts = make_prompts(args.batch, args.prompt_words)

    tok = llm.get_tokenizer()
    prompt_tokens_each = len(tok.encode(prompts[0]))

    for _ in range(args.warmup_runs):
        llm.generate(make_prompts(args.batch, args.prompt_words),
                     sampling, use_tqdm=False)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    seconds_total = time.perf_counter() - t0

    produced = sum(len(o.token_ids)
                   for out in outputs for o in out.outputs)

    result = {
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "prompt_tokens_each": prompt_tokens_each,
        "seconds_total": round(seconds_total, 4),
        "seconds_per_inference": round(seconds_total / args.batch, 6),
        "output_tokens_total": produced,
        "output_tokens_per_second": round(produced / seconds_total, 2),
        "status": "ok",
        "error": "",
    }
    # single tagged line, so it survives a truncated log
    print("QVRESULT " + json.dumps(result))


# ----------------------------------------------------------------------------
# the driver
# ----------------------------------------------------------------------------

def child_command(args, model, quantization, enforce_eager, batch):
    cmd = [
        sys.executable, os.path.abspath(__file__), "--single",
        "--model", model,
        "--dtype", args.dtype,
        "--batch", str(batch),
        "--max-tokens", str(args.max_tokens),
        "--prompt-words", str(args.prompt_words),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--warmup-runs", str(args.warmup_runs),
        "--seed", str(args.seed),
    ]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if quantization:
        cmd += ["--quantization", quantization]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    return cmd


def spread_report(rows):
    """Per-configuration repeatability, when --repeats > 1."""
    import statistics as st

    groups = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = (r["model"], r["quantization"], r["enforce_eager"], r["batch"])
        groups.setdefault(key, []).append(r["seconds_per_inference"])

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        return

    print("")
    print("REPEATABILITY  (spread across repeats of the identical config)")
    print("  %-38s %10s %8s %8s" % ("config", "mean s", "cv", "spread"))
    for k in sorted(multi, key=lambda x: (x[0], x[2], x[3])):
        model, quant, eager, batch = k
        v = multi[k]
        mean = st.mean(v)
        cv = (st.stdev(v) / mean * 100) if mean else 0.0
        spread = (max(v) / min(v) - 1) * 100 if min(v) else 0.0
        label = "%s %s b%d" % (quant, "eager" if eager else "cuda-graphs", batch)
        print("  %-38s %10.6f %7.2f%% %7.2f%%" % (label[:38], mean, cv, spread))
    print("")
    print("  A config whose own repeats disagree by more than a few percent cannot")
    print("  support a ratio claim smaller than that. Report the spread, not one run.")


def drive(args):
    env = probe_gpus()
    env.update(probe_cpu())
    run_id = uuid.uuid4().hex[:12]

    if env["gpu_count"] == 0:
        print("No GPU visible to nvidia-smi. This needs a CUDA GPU.")
        return 1

    # a "variant" is one model build. Two builds of the same weights in
    # different formats are two variants, because they are two model ids.
    variants = [(args.model, "")]
    if args.compare_model:
        variants.append((args.compare_model, args.compare_quantization))

    batches = [int(b) for b in args.batches.split(",")]
    modes = []
    if "graphs" in args.modes:
        modes.append(False)   # enforce_eager = False
    if "eager" in args.modes:
        modes.append(True)

    configs = [(m, q, e, b)
               for (m, q) in variants for e in modes for b in batches]
    total = len(configs) * args.repeats

    print("qvunex sweep")
    print("  gpu        : %s x%d (driver %s)"
          % (env["gpu_name"], env["gpu_count"], env["driver_version"]))
    print("  cpu        : %s (%d cores visible)"
          % (env["cpu_model"], env["cpu_cores"]))
    for m, q in variants:
        print("  model      : %s%s" % (m, ("  [%s]" % q) if q else ""))
    print("  configs    : %d x %d repeat(s) = %d runs"
          % (len(configs), args.repeats, total))
    print("  output     : %s" % args.out)
    print("")

    rows = []
    n = 0
    # repeats on the outside, so a slow drift in the machine spreads across
    # every config instead of landing entirely on one of them
    for rep in range(args.repeats):
        for (model, quant, eager, batch) in configs:
            n += 1
            label = "%s / %s / batch %d" % (
                quant or args.dtype, "eager" if eager else "cuda-graphs", batch)
            suffix = "  (repeat %d/%d)" % (rep + 1, args.repeats) \
                if args.repeats > 1 else ""
            print("[%d/%d] %s%s" % (n, total, label, suffix), flush=True)

            row = dict.fromkeys(FIELDS, "")
            row.update(env)
            row.update({
                "run_id": run_id,
                "repeat": rep,
                "timestamp_utc": datetime.now(timezone.utc)
                                         .isoformat(timespec="seconds"),
                "model": model,
                "quantization": quant or "none",
                "dtype": args.dtype,
                "enforce_eager": eager,
                "prefix_caching": False,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len or "",
                "tensor_parallel_size": args.tensor_parallel_size,
                "batch": batch,
                "max_tokens": args.max_tokens,
                "ignore_eos": True,
                "seed": args.seed,
                "warmup_runs": args.warmup_runs,
            })

            proc = subprocess.run(
                child_command(args, model, quant, eager, batch),
                capture_output=True, text=True)

            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith("QVRESULT "):
                    payload = json.loads(line[len("QVRESULT "):])

            if payload:
                row.update(payload)
                print("        %.4f s total  ->  %.6f s per inference"
                      % (row["seconds_total"], row["seconds_per_inference"]),
                      flush=True)
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()
                row["status"] = "failed"
                row["error"] = (tail[-1] if tail else "no output")[:300]
                print("        FAILED: %s" % row["error"], flush=True)

            rows.append(row)

            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print("")
    print("wrote %s  (%d ok, %d failed)"
          % (args.out, len(ok), len(rows) - len(ok)))

    if len(ok) > 1:
        best = min(ok, key=lambda r: r["seconds_per_inference"])
        worst = max(ok, key=lambda r: r["seconds_per_inference"])
        spread = worst["seconds_per_inference"] / best["seconds_per_inference"]
        print("")
        print("cheapest : %.6f s  (%s, %s, batch %d)"
              % (best["seconds_per_inference"], best["quantization"],
                 "eager" if best["enforce_eager"] else "cuda-graphs",
                 best["batch"]))
        print("dearest  : %.6f s  (%s, %s, batch %d)"
              % (worst["seconds_per_inference"], worst["quantization"],
                 "eager" if worst["enforce_eager"] else "cuda-graphs",
                 worst["batch"]))
        print("spread   : %.1fx on one model, one GPU" % spread)

    spread_report(ok)

    print("")
    print("If you are comparing this against someone else's number, send them the")
    print("whole CSV. Every column in it is a reason two numbers can disagree.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--compare-model", default=None,
                   help="a second model id to sweep alongside --model, e.g. an "
                        "AWQ build of the same weights. Quantized formats are "
                        "separate model ids, not a flag on one model.")
    p.add_argument("--compare-quantization", default="",
                   help="quantization backend for --compare-model, e.g. awq, "
                        "gptq, fp8. Leave empty to let vLLM detect it.")
    p.add_argument("--batches", default="1,8,32,128")
    p.add_argument("--modes", default="eager,graphs")
    p.add_argument("--repeats", type=int, default=1,
                   help="run the whole sweep this many times, one row per "
                        "repeat. Use 3 or more before publishing a ratio.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-words", type=int, default=64)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="qvunex-sweep.csv")

    # internal: run exactly one configuration in this process
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--batch", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--quantization", default="", help=argparse.SUPPRESS)
    p.add_argument("--enforce-eager", action="store_true", help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.single:
        run_single(args)
        return 0
    return drive(args)


if __name__ == "__main__":
    # vLLM's V1 engine child handshake fails in some containers; this keeps the
    # engine in-process and is what made these runs reproducible for us.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.exit(main())
