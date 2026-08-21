"""A synthetic workload, so the report can be seen before wiring up a real one.

Deliberately shaped like a workload with real problems: one dominant endpoint,
a cheap one that runs constantly, ragged batch sizes, and idle gaps. Those are
the things the report is meant to surface, so the demo should contain them.
"""

import random
import time

from . import meter
from .core import configure, flush, _config
from .report import analyse, render


def run_demo(seconds=12.0, rate=0.35):
    configure(path="./qvunex_demo.jsonl", rate_usd_hour=rate, sample_interval=0.5)

    @meter("image-classifier")
    def classify(batch):
        time.sleep(0.010 + 0.0015 * len(batch))
        return [0] * len(batch)

    @meter("text-embedder")
    def embed(batch):
        time.sleep(0.004 * len(batch))
        return [[0.0]] * len(batch)

    @meter("thumbnail-scorer")
    def score(batch):
        time.sleep(0.001)
        return [0.5] * len(batch)

    rng = random.Random(7)
    t_end = time.time() + seconds
    print(f"running synthetic workload for {seconds:.0f}s ...")

    while time.time() < t_end:
        # ragged batches -- the batch-efficiency signal
        classify([0] * rng.choice([32, 8, 4, 32, 2, 16]))
        embed([0] * rng.choice([16, 4, 8]))
        for _ in range(3):
            score([0])
        # a quiet gap -- the idle-burn signal
        if rng.random() < 0.25:
            time.sleep(rng.uniform(0.3, 0.9))

    flush()
    print()
    print(render(analyse(_config["path"], rate)))
    print(f"raw events: {_config['path']}")
    return 0
