"""Minimal test runner (pytest unavailable in this sandbox)."""
import sys, os, traceback
sys.path.insert(0, "src"); sys.path.insert(0, "tests")
import test_meter as T
import qvunex.core as M

def reset():
    if M._session is not None:
        try: M._session.close()
        except Exception: pass
    M._session = None
    M._config.update({"path": M.DEFAULT_PATH, "rate_usd_hour": 0.0,
                      "sample_interval": 1.0, "enabled": True})

names = sorted(n for n in dir(T) if n.startswith("test_"))
passed = failed = 0
for n in names:
    reset()
    try:
        getattr(T, n)()
        print(f"  PASS  {n}"); passed += 1
    except Exception:
        print(f"  FAIL  {n}")
        traceback.print_exc(limit=3)
        failed += 1
reset()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
