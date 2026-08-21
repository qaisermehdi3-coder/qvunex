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
"""

SCHEMA_VERSION = "0.1"

CALL = "call"
GPU = "gpu"
SESSION = "session"


def session_record(ts, host, gpus, config):
    return {
        "t": SESSION,
        "v": SCHEMA_VERSION,
        "ts": ts,
        "host": host,
        "gpus": gpus,
        "config": config,
    }


def call_record(ts, endpoint, dur_ms, batch, ok, error=None, meta=None):
    r = {
        "t": CALL,
        "v": SCHEMA_VERSION,
        "ts": ts,
        "endpoint": endpoint,
        "dur_ms": dur_ms,
        "batch": batch,
        "ok": ok,
    }
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
