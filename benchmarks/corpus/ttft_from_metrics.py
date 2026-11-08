#!/usr/bin/env python3
"""Mean time to first token for one load level, from vLLM's own /metrics.

  python3 ttft_from_metrics.py before_64.txt after_64.txt

Reads the vllm:time_to_first_token_seconds histogram in the two /metrics
snapshots that planb.sh saves before and after each level, and prints
(sum after - sum before) / (count after - count before), and that count.
The count should equal requests_done for the level. Standard library only.
"""
import re
import sys


def read(path):
    s = c = None
    for line in open(path):
        m = re.match(r'vllm:time_to_first_token_seconds_(sum|count)\{[^}]*\} ([0-9.e+]+)\s*$', line)
        if m:
            if m.group(1) == "sum":
                s = (s or 0) + float(m.group(2))
            else:
                c = (c or 0) + float(m.group(2))
    return s, c


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (bs, bc), (as_, ac) = read(sys.argv[1]), read(sys.argv[2])
    if None in (bs, bc, as_, ac) or ac - bc <= 0:
        print("no time-to-first-token histogram found (unknown, not zero)")
        return 1
    print(f"mean time to first token {(as_ - bs) / (ac - bc):.3f} s over {ac - bc:.0f} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
