"""Qvunex — measure what your inference actually costs.

    from qvunex import meter

    @meter("checkout-classifier")
    def predict(batch):
        return model(batch)

Then, from a shell:

    qvunex report

Everything is written to a local file. This package contains no network code.
"""

from .core import configure, flush, meter
from .report import analyse, render
from .schema import SCHEMA_VERSION

__version__ = "0.2.1"
__all__ = ["meter", "configure", "flush", "analyse", "render", "SCHEMA_VERSION"]
