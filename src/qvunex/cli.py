"""`qvunex report` and friends."""

import argparse
import json
import sys

from .core import DEFAULT_PATH
from .report import analyse, render
from .store import Store


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="qvunex",
        description="Measure what your inference actually costs. Local only.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("report", help="summarise a corpus file")
    r.add_argument("path", nargs="?", default=DEFAULT_PATH)
    r.add_argument("--rate", type=float, default=None,
                   metavar="USD_PER_HOUR",
                   help="GPU price per hour; overrides the recorded config")
    r.add_argument("--idle-threshold", type=float, default=5.0,
                   help="utilisation %% below which a GPU counts as idle")
    r.add_argument("--json", action="store_true",
                   help="emit the raw analysis instead of the text report")

    c = sub.add_parser("checklist",
                       help="emit a Comparability Checklist for a corpus file")
    c.add_argument("path", nargs="?", default=DEFAULT_PATH)

    d = sub.add_parser("demo", help="run a synthetic workload and report on it")
    d.add_argument("--seconds", type=float, default=12.0)
    d.add_argument("--rate", type=float, default=0.35)

    args = p.parse_args(argv)

    if args.cmd == "report":
        a = analyse(args.path, args.rate, args.idle_threshold)
        if args.json:
            print(json.dumps(a, indent=2, default=str))
        else:
            print(render(a))
        return 0

    if args.cmd == "checklist":
        from .checklist import build
        ctx = {}
        for r in Store.read(args.path):
            if r.get("t") == "session":
                ctx.update((r.get("config") or {}).get("context") or {})
        lines, missing = build(args.path, ctx)
        print("\n".join(lines))
        return 1 if missing else 0

    if args.cmd == "demo":
        from .demo import run_demo
        return run_demo(args.seconds, args.rate)

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
