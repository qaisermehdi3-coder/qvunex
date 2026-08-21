"""Background GPU sampling via nvidia-smi.

Deliberately shells out to nvidia-smi rather than depending on pynvml or torch.
The meter must run inside someone else's environment without adding a dependency
they have to approve, and nvidia-smi is present wherever an NVIDIA GPU is.

Absence of a GPU is a normal state, not an error: the sampler reports
`available == False` and the report degrades to the metrics that don't need it.
"""

import subprocess
import threading
import time

from .schema import gpu_record

_QUERY = "index,utilization.gpu,power.draw,memory.used,memory.total"
_INFO = "index,name,driver_version,power.limit,memory.total"


def _smi(query, extra=()):
    cmd = ["nvidia-smi", f"--query-gpu={query}",
           "--format=csv,noheader,nounits", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200] or "nvidia-smi failed")
    return [l.strip() for l in out.stdout.strip().split("\n") if l.strip()]


def gpu_info():
    """Static device description, or [] when there is no GPU."""
    try:
        rows = _smi(_INFO)
    except Exception:
        return []
    out = []
    for row in rows:
        p = [x.strip() for x in row.split(",")]
        if len(p) < 5:
            continue
        try:
            out.append({
                "index": int(p[0]), "name": p[1], "driver": p[2],
                "power_limit_w": float(p[3]), "memory_total_mib": float(p[4]),
            })
        except ValueError:
            continue
    return out


class Sampler:
    """Polls device state on a background thread and writes it to the store."""

    def __init__(self, store, interval=1.0):
        self.store = store
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.available = bool(gpu_info())

    def _sample_once(self):
        try:
            rows = _smi(_QUERY)
        except Exception:
            return
        ts = time.time()
        for row in rows:
            p = [x.strip() for x in row.split(",")]
            if len(p) < 5:
                continue
            try:
                self.store.write(gpu_record(
                    ts, int(p[0]), float(p[1]), float(p[2]),
                    float(p[3]), float(p[4]),
                ))
            except ValueError:
                # "[N/A]" appears for power.draw on some consumer cards.
                continue

    def _loop(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def start(self):
        if not self.available or self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="qvunex-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval + 2)
        self._thread = None
