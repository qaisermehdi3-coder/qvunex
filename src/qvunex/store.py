"""Local-only event storage.

Everything Qvunex records is appended to a JSONL file on the user's own disk.
Nothing is transmitted anywhere — there is no network code in this package, by
design. That is a strategic choice, not a limitation: a read-only tool that never
phones home can be adopted in an afternoon instead of surviving a six-month
security review.

Writes are buffered and flushed on a size/time threshold so that instrumenting a
hot inference path costs microseconds, not a syscall per call.
"""

import json
import os
import threading
import time


class Store:
    def __init__(self, path, flush_every=200, flush_seconds=5.0):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._buf = []
        self._lock = threading.Lock()
        self._flush_every = flush_every
        self._flush_seconds = flush_seconds
        self._last_flush = time.time()
        self._closed = False

    def write(self, record):
        if self._closed:
            return
        with self._lock:
            self._buf.append(record)
            due = (
                len(self._buf) >= self._flush_every
                or (time.time() - self._last_flush) >= self._flush_seconds
            )
            if due:
                self._flush_locked()

    def _flush_locked(self):
        if not self._buf:
            return
        lines = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self._buf)
        self._buf = []
        self._last_flush = time.time()
        # Written outside the buffer swap would be nicer, but the append is fast
        # and holding the lock keeps ordering deterministic.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(lines)

    def flush(self):
        with self._lock:
            self._flush_locked()

    def close(self):
        self.flush()
        self._closed = True

    # ---- reading -------------------------------------------------------

    @staticmethod
    def read(path):
        """Yield records from a JSONL file, skipping any corrupt trailing line.

        A partial final line is normal if a process was killed mid-flush; it is
        skipped rather than raising, so a crashed run still yields a usable report.
        """
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
