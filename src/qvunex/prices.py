"""Turn recorded tokens into money, or say plainly that it cannot.

The meter records what happened. This file is the only place that knows what any
of it costs, and it knows nothing until you tell it. There is no built-in rate
card and no default price, deliberately: rate cards change without notice, a
stale one is worse than none, and a cost report that quietly invents a number is
the one failure this project cannot come back from.

A call whose model has no entry is reported as **unpriced**. It is still counted,
its tokens still show up, and any task containing it is flagged as partial rather
than given a smaller-than-true number.

The rate card is a plain text file, the same shape `tools/taskcost.py` uses:

    price: claude-sonnet-4-5
      input_per_mtok: 3.00
      output_per_mtok: 15.00
      cache_write_per_mtok: 3.75     # optional; defaults to 1.25x input
      cache_read_per_mtok: 0.30      # optional; defaults to 0.10x input

Two details that are easy to get wrong and expensive to get wrong:

**A cache write costs more than not caching.** 1.25x the input rate on a 5-minute
TTL, 2x on an hour. A read costs 0.1x. Treating cache tokens as ordinary input
makes a cold run and a warm run look identical when they can differ several fold.
The defaults here assume the 5-minute TTL, because that is the default TTL; if
you use the hour, set `cache_write_per_mtok` yourself rather than letting this
file guess on your behalf. Whichever applies, the assumption is printed.

**Reasoning tokens are already inside the output count.** Both providers that
report them separately report them as a breakdown of tokens they have already
billed, not as an extra line. They are recorded so you can see them; they are not
added again here. Adding them would inflate every reasoning-heavy task by roughly
a third, which is the same error as ignoring them, pointed the other way.

Model ids carry a date suffix the rate card does not. Matching is by longest
prefix, so one `claude-sonnet-4-5` entry covers every dated build of it.
"""

import os

DEFAULT_PRICES_PATH = "~/.qvunex/prices.txt"

# Multipliers on the base input rate, used only when the card does not state a
# cache price outright. Both are printed in the report so nobody has to read this
# file to find out what was assumed.
CACHE_WRITE_MULTIPLIER_5M = 1.25
CACHE_READ_MULTIPLIER = 0.10

_FIELDS = {
    "input_per_mtok": "input",
    "output_per_mtok": "output",
    "cache_write_per_mtok": "cache_write",
    "cache_read_per_mtok": "cache_read",
}


def parse(text):
    """Parse a rate card. Returns {model_prefix: {input, output, ...}}.

    Unknown keys and malformed lines are skipped rather than raising: a typo in
    one stanza should cost you that stanza, not the whole report.
    """
    cards = {}
    current = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("price:"):
            name = line.split(":", 1)[1].strip()
            if name:
                current = cards.setdefault(name, {})
            continue
        if current is None or ":" not in line:
            continue
        key, _, value = line.strip().partition(":")
        field = _FIELDS.get(key.strip())
        if field is None:
            continue
        try:
            current[field] = float(value.strip())
        except ValueError:
            continue
    return {k: v for k, v in cards.items() if v.get("input") is not None
            or v.get("output") is not None}


def load(path=None):
    """Read a rate card from disk. Returns {} when there isn't one."""
    path = os.path.expanduser(path or DEFAULT_PRICES_PATH)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return parse(fh.read())
    except OSError:
        return {}


def card_for(model, cards):
    """Longest matching prefix, so a dated model id finds its undated entry."""
    if not model or not cards:
        return None
    best = None
    for prefix, card in cards.items():
        if model.startswith(prefix) or prefix.startswith(model):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, card)
    return best[1] if best else None


def rates(card):
    """Fill in cache rates a card leaves out, from the input rate."""
    inp = card.get("input")
    out = dict(card)
    if inp is not None:
        out.setdefault("cache_write", inp * CACHE_WRITE_MULTIPLIER_5M)
        out.setdefault("cache_read", inp * CACHE_READ_MULTIPLIER)
    return out


def cost_of(record, cards):
    """Cost of one recorded call in USD, or None if it cannot be priced.

    None means "not known", never zero. A call with no token counts at all — a
    @meter'd local function, say — is also None here; local GPU time is costed
    the other way, by rate_usd_hour and wall clock.
    """
    has_tokens = any(k in record for k in
                     ("tokens_in", "tokens_out", "cache_read", "cache_write"))
    if not has_tokens:
        return None
    card = card_for(record.get("model"), cards)
    if not card:
        return None
    r = rates(card)
    total = 0.0
    for field, rate_key in (("tokens_in", "input"),
                            ("tokens_out", "output"),
                            ("cache_write", "cache_write"),
                            ("cache_read", "cache_read")):
        n = record.get(field)
        if not n:
            continue
        rate = r.get(rate_key)
        if rate is None:
            return None          # priced in part is not priced
        total += n / 1_000_000.0 * rate
    # tokens_reasoning is deliberately absent: it is a breakdown of output
    # tokens already counted above, not an additional charge.
    return total
