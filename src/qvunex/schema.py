"""Event schema for the Qvunex corpus.

Blueprint Rule #3: normalise and keep every measurement from day one. The corpus
cannot be collected retroactively, so the schema is fixed here and versioned, not
invented per-customer.

Three record types, one JSON object per line:

  session  written once at start; environment + config
  call     one per instrumented inference call
  gpu      one per background sample of device state

Every record carries `t` (type), `ts` (unix seconds, float) and `v` (schema
version). Additive changes bump the minor version; anything that changes the
meaning of an existing field bumps the major and gets a migration note here.

Version history
  0.1  initial — call/gpu/session
  0.2  call gains the fields needed to answer "what did one finished task cost":
       task, model, token counts split by kind, and status. Additive — a 0.1
       reader sees a 0.2 call record with extra keys, and the old fields keep
       their meaning. The field list came from practitioners rather than from
       us: per-request tokens split into input, output and cache, a model id, a
       route or task tag, wall time, and success-versus-retry-versus-fallback as
       a status. Without the tag and the cache split, cost reports lie even when
       the totals look neat.
"""

SCHEMA_VERSION = "0.2"

CALL = "call"
GPU = "gpu"
SESSION = "session"

# A call is one of these. `retry` and `fallback` are real spend that ordinary
# logging drops on the floor: a retry is the same work paid for twice, and a
# fallback is a different, usually dearer, model quietly doing the job.
OK = "ok"
FAILED = "failed"
RETRY = "retry"
FALLBACK = "fallback"

TOKEN_FIELDS = ("tokens_in", "tokens_out", "tokens_reasoning",
                "cache_read", "cache_write")


def session_record(ts, host, gpus, config):
    return {
        "t": SESSION,
        "v": SCHEMA_VERSION,
        "ts": ts,
        "host": host,
        "gpus": gpus,
        "config": config,
    }


def call_record(ts, endpoint, dur_ms, batch, ok, error=None, meta=None,
                task=None, task_id=None, model=None, status=None, usage=None,
                attempt=None):
    """One inference call.

    ok/error/dur_ms/batch are unchanged from 0.1.

    task / task_id  the unit of work this call belongs to. Without an id
                    threaded through every call a task makes — including the
                    ones a framework makes on your behalf — per-task cost cannot
                    be reconstructed afterwards, only guessed at.
    model           what actually answered, which is not always what you asked
                    for once a fallback fires.
    status          ok / failed / retry / fallback.
    usage           token counts from usage.extract(); absent keys mean the
                    provider did not report that number, which is different from
                    reporting zero.
    attempt         1 for the first try, 2+ for retries of the same logical call.
    """
    r = {
        "t": CALL,
        "v": SCHEMA_VERSION,
        "ts": ts,
        "endpoint": endpoint,
        "dur_ms": dur_ms,
        "batch": batch,
        "ok": ok,
        "status": status or (OK if ok else FAILED),
    }
    if task:
        r["task"] = task
    if task_id:
        r["task_id"] = task_id
    if model:
        r["model"] = model
    if attempt and attempt != 1:
        r["attempt"] = int(attempt)
    if usage:
        for key in TOKEN_FIELDS:
            if key in usage and usage[key] is not None:
                r[key] = usage[key]
    if error:
        r["error"] = error
    if meta:
        r["meta"] = meta
    return r


def gpu_record(ts, index, util, power_w, mem_used_mib, mem_total_mib):
    return {
        "t": GPU,
        "v": SCHEMA_VERSION,
        "ts": ts,
        "index": index,
        "util": util,
        "power_w": power_w,
        "mem_used_mib": mem_used_mib,
        "mem_total_mib": mem_total_mib,
    }
