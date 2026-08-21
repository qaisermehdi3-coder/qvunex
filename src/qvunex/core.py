"""The meter — the two lines a user adds to their own code.

    from qvunex import meter

    @meter("checkout-classifier")
    def predict(batch):
        return model(batch)

That is the entire integration surface. Everything else (the background sampler,
the session record, flushing at exit) starts itself on first use.

Design constraints, in priority order:

1. **Never break the caller.** Any failure inside the meter is swallowed; the
   wrapped function's exception propagates untouched, and a call that raises is
   still recorded with ok=False. A measurement tool that can take down production
   is a tool nobody installs twice.
2. **Cheap.** The hot path is a perf_counter pair, a dict, and an append to a
   buffered list. No I/O per call.
3. **Local only.** No network. See store.py.
"""

import atexit
import functools
import os
import socket
import threading
import time

from .sampler import Sampler, gpu_info
from .schema import call_record, session_record
from .store import Store

DEFAULT_PATH = "~/.qvunex/events.jsonl"

_config = {
    "path": os.environ.get("QVUNEX_PATH", DEFAULT_PATH),
    "rate_usd_hour": float(os.environ.get("QVUNEX_RATE_USD_HOUR", "0") or 0),
    "sample_interval": float(os.environ.get("QVUNEX_SAMPLE_INTERVAL", "1.0")),
    "enabled": os.environ.get("QVUNEX_DISABLED", "").lower() not in ("1", "true", "yes"),
    "context": {},
}

_session = None
_session_lock = threading.Lock()


def configure(path=None, rate_usd_hour=None, sample_interval=None, enabled=None,
              context=None):
    """Set options before first use. Env vars are the defaults.

    rate_usd_hour is what you pay per GPU-hour. Without it the report still shows
    utilisation, latency and idle time; it just cannot convert them to money.

    context is what the meter cannot observe for itself — engine version, whether
    CUDA graphs are on, which model is loaded. It is recorded verbatim in the
    session and used to fill the Comparability Checklist. Anything omitted here
    is printed as MISSING rather than quietly dropped.
    """
    if _session is not None:
        raise RuntimeError("qvunex.configure() must be called before the first metered call")
    if path is not None:
        _config["path"] = path
    if rate_usd_hour is not None:
        _config["rate_usd_hour"] = float(rate_usd_hour)
    if sample_interval is not None:
        _config["sample_interval"] = float(sample_interval)
    if enabled is not None:
        _config["enabled"] = bool(enabled)
    if context is not None:
        _config["context"] = dict(context)


class _Session:
    def __init__(self):
        self.store = Store(_config["path"])
        self.sampler = Sampler(self.store, _config["sample_interval"])
        self.started = time.time()
        self.store.write(session_record(
            self.started,
            socket.gethostname(),
            gpu_info(),
            {k: v for k, v in _config.items() if k != "enabled"},
        ))
        self.sampler.start()
        atexit.register(self.close)

    def close(self):
        try:
            self.sampler.stop()
            self.store.close()
        except Exception:
            pass


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = _Session()
    return _session


def flush():
    """Force pending records to disk. Called automatically at process exit."""
    if _session is not None:
        _session.store.flush()


def _infer_batch(args, kwargs):
    """Best-effort batch size from the first argument.

    Handles torch/numpy (.shape[0]), sequences (len), and falls back to 1. Wrong
    is better than raising here — the caller's code must never break because we
    could not guess a batch size.
    """
    obj = args[0] if args else next(iter(kwargs.values()), None)
    if obj is None:
        return 1
    try:
        shape = getattr(obj, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    except Exception:
        pass
    try:
        if isinstance(obj, (str, bytes)):
            return 1
        return max(1, len(obj))
    except Exception:
        return 1


class _Span:
    """Context manager form, for code that isn't shaped like a function."""

    def __init__(self, endpoint, batch=1, meta=None):
        self.endpoint = endpoint
        self.batch = batch
        self.meta = meta
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dur_ms = (time.perf_counter() - self._t0) * 1000.0
        _record(self.endpoint, dur_ms, self.batch, exc is None,
                None if exc is None else type(exc).__name__, self.meta)
        return False


def _record(endpoint, dur_ms, batch, ok, error=None, meta=None):
    if not _config["enabled"]:
        return
    try:
        _get_session().store.write(call_record(
            time.time(), endpoint, round(dur_ms, 4), batch, ok, error, meta))
    except Exception:
        pass  # never let instrumentation break the caller


def meter(endpoint, batch=None, meta=None):
    """Decorate an inference function so every call is recorded.

    batch: int, or a callable (args, kwargs) -> int, when the batch size cannot be
    inferred from the first argument.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _config["enabled"]:
                return fn(*args, **kwargs)
            if callable(batch):
                try:
                    n = int(batch(args, kwargs))
                except Exception:
                    n = 1
            elif isinstance(batch, int):
                n = batch
            else:
                n = _infer_batch(args, kwargs)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                _record(endpoint, (time.perf_counter() - t0) * 1000.0,
                        n, False, type(e).__name__, meta)
                raise
            _record(endpoint, (time.perf_counter() - t0) * 1000.0,
                    n, True, None, meta)
            return out
        return wrapper
    return decorator


meter.span = _Span
