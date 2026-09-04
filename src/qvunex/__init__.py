"""Qvunex — measure what your inference actually costs.

Time your own functions:

    from qvunex import meter

    @meter("checkout-classifier")
    def predict(batch):
        return model(batch)

Or record every call through a provider client, including the ones your
framework makes on your behalf:

    from qvunex import wrap, task

    client = wrap(anthropic.Anthropic())

    with task("draft outreach email"):
        ...

Then, from a shell:

    qvunex report

Everything is written to a local file. This package contains no network code.
"""

from .core import (configure, current_task, flush, meter, record_retry, task,
                   wrap)
from .report import analyse, render
from .schema import SCHEMA_VERSION

__version__ = "0.3.0"
__all__ = ["meter", "task", "wrap", "current_task", "record_retry",
           "configure", "flush", "analyse", "render", "SCHEMA_VERSION"]
