"""The meter — the lines a user adds to their own code.

Three ways in, smallest first:

    from qvunex import meter, task, wrap

    @meter("checkout-classifier")          # time a function
    def predict(batch):
        return model(batch)

    client = wrap(anthropic.Anthropic())   # record every call through a client

    with task("draft outreach email"):     # group calls into one unit of work
        ...

`wrap` matters more than it looks. Decorating your own functions only sees the
calls you wrote. Wrapping the client also catches the calls your framework makes
on your behalf — which is where the spend hides when one request fans out into
six sub-agent calls and the usage you can read back reports the last one only.

`task` is the correlation id. Per-task cost cannot be reconstructed after the
fact from per-call billing data; the id has to be attached at call time or the
number is a guess.

Design constraints, in priority order:

1. **Never break the caller.** Any failure inside the meter is swallowed; the
   wrapped function's exception propagates untouched, and a call that raises is
   still recorded. A measurement tool that can take down production is a tool
   nobody installs twice.
2. **Cheap.** The hot path is a perf_counter pair, a dict, and an append to a
   buffered list. No I/O per call.
3. **Local only.** No network. See store.py.

Known limits, stated rather than hidden:

* If your SDK retries internally, a client wrapper sits above those retries and
  sees one call, not three. Set the SDK's own max_retries to 0 and retry in your
  code if you need them counted.
* A streaming response has no usage attached when the call returns, so tokens
  are recorded only for non-streaming calls.
"""

import atexit
import contextvars
import functools
import os
import socket
import threading
import time
import uuid

from .sampler import Sampler, gpu_info
from .schema import FALLBACK, RETRY, call_record, session_record
from .store import Store
from .usage import extract

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

# The correlation id, carried through nested calls and across threads that
# inherit the context. None when a call happens outside any task.
_task_var = contextvars.ContextVar("qvunex_task", default=None)


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


# ---------------------------------------------------------------------------
# tasks — the correlation id
# ---------------------------------------------------------------------------

class _Task:
    """One unit of finished work. Every call inside it carries its id.

        with task("draft outreach email"):
            classify(...)      # these three calls
            research(...)      # all belong to
            draft(...)         # one task

    Nests: an inner task replaces the id for its own body and restores the outer
    one on exit.
    """

    def __init__(self, name, task_id=None):
        self.name = name
        self.id = task_id or uuid.uuid4().hex[:12]
        self._token = None

    def __enter__(self):
        self._token = _task_var.set((self.name, self.id))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _task_var.reset(self._token)
        return False


def task(name, task_id=None):
    """Group everything inside into one finished task. See _Task."""
    return _Task(name, task_id)


def current_task():
    """(name, id) of the task in progress, or None."""
    return _task_var.get()


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


def _record(endpoint, dur_ms, batch, ok, error=None, meta=None,
            usage=None, model=None, status=None, attempt=None):
    if not _config["enabled"]:
        return
    try:
        current = _task_var.get()
        name, tid = current if current else (None, None)
        _get_session().store.write(call_record(
            time.time(), endpoint, round(dur_ms, 4), batch, ok, error, meta,
            task=name, task_id=tid, model=model, status=status, usage=usage,
            attempt=attempt))
    except Exception:
        pass  # never let instrumentation break the caller


def _usage_of(value):
    """Token counts off a provider response, or None. Never raises."""
    try:
        return extract(value)
    except Exception:
        return None


def meter(endpoint, batch=None, meta=None):
    """Decorate an inference function so every call is recorded.

    batch: int, or a callable (args, kwargs) -> int, when the batch size cannot be
    inferred from the first argument.

    If the function returns a provider response, its token counts are recorded
    too, so a decorated function that calls an API gets the same detail as a
    wrapped client.
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
            u = _usage_of(out)
            _record(endpoint, (time.perf_counter() - t0) * 1000.0,
                    n, True, None, meta,
                    usage=u, model=(u or {}).get("model"))
            return out
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# wrapping a provider client
# ---------------------------------------------------------------------------

# Where the call actually happens on the common clients. Duck-typed: a client
# that doesn't have one of these paths is skipped rather than erroring.
_CALL_PATHS = (
    ("messages", "create"),             # Anthropic
    ("beta", "messages", "create"),     # Anthropic beta
    ("chat", "completions", "create"),  # OpenAI chat
    ("responses", "create"),            # OpenAI responses
    ("completions", "create"),          # OpenAI legacy
)


def _wrapped_call(fn, endpoint):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        if not _config["enabled"]:
            return fn(*args, **kwargs)
        route = kwargs.get("qvunex_route")
        if route is not None:
            kwargs = {k: v for k, v in kwargs.items() if k != "qvunex_route"}
        name = route or endpoint
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            _record(name, (time.perf_counter() - t0) * 1000.0, 1, False,
                    type(e).__name__)
            raise
        u = _usage_of(out)
        asked = kwargs.get("model")
        answered = (u or {}).get("model")
        status = FALLBACK if (asked and answered
                              and asked not in answered) else None
        _record(name, (time.perf_counter() - t0) * 1000.0, 1, True,
                usage=u, model=answered or asked, status=status)
        return out

    call._qvunex_wrapped = True
    return call


def wrap(client, endpoint=None):
    """Record every call made through a provider client, and return it.

        client = wrap(anthropic.Anthropic())

    Pass qvunex_route="my-step" to any call to label that one route; the keyword
    is stripped before the provider sees it.

    A call whose response reports a different model than the one asked for is
    recorded as a fallback — real spend that ordinary logging attributes to the
    model you thought you were using.

    Returns the same client object, mutated in place, so it is safe to wrap once
    at startup and pass the client around as normal.
    """
    for path in _CALL_PATHS:
        holder = client
        for part in path[:-1]:
            holder = getattr(holder, part, None)
            if holder is None:
                break
        if holder is None:
            continue
        name = path[-1]
        fn = getattr(holder, name, None)
        if fn is None or getattr(fn, "_qvunex_wrapped", False):
            continue
        try:
            setattr(holder, name, _wrapped_call(fn, endpoint or ".".join(path)))
        except Exception:
            pass  # some clients forbid attribute assignment; skip that path
    return client


def record_retry(endpoint, dur_ms=0.0, attempt=2, model=None, usage=None):
    """Record an attempt you retried yourself.

    Only needed when you handle retries in your own code. Spend on a retried
    call is real, it is on your bill, and it is invisible to anything that only
    records the attempt that finally succeeded.
    """
    _record(endpoint, dur_ms, 1, True, usage=usage, model=model,
            status=RETRY, attempt=attempt)


meter.span = _Span
