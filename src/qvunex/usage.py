"""Read token counts off a provider's response object.

Every provider reports usage in a slightly different shape, and the differences
are not cosmetic. Two of them matter enough to be the reason this file exists:

**Cache tokens are not a discount, they are separate line items.** A cache write
costs *more* than not caching at all (1.25x input price on a 5-minute TTL, 2x on
an hour); a cache read costs 0.1x. Folding them into one "input tokens" number
makes a cold run and a warm run look identical when they differ by an order of
magnitude.

**Reasoning tokens are billed but rarely broken out.** On reasoning-heavy work
they routinely exceed the visible output. Providers bill them as input or as
output depending on the provider, and almost never as their own line, so a
dashboard that surfaces completion tokens alone undercounts spend by roughly a
third to a half. Where a provider does report them, we record them separately.
Where it doesn't — Anthropic folds thinking into output_tokens — we say so rather
than guess.

No dependencies and no imports of any provider SDK. Everything here is
duck-typed, so a new provider that reports the same field names works without a
change, and one that doesn't returns None instead of raising.
"""


def _get(obj, name):
    """Read `name` off an object or a dict, or None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract(response):
    """Return a usage dict for a provider response, or None if there isn't one.

    Keys, all optional and all integers when present:

        tokens_in         non-cached input tokens billed at the base rate
        tokens_out        output tokens
        tokens_reasoning  thinking tokens, when the provider reports them apart
        cache_write       tokens written to the prompt cache (dearer than input)
        cache_read        tokens served from the prompt cache (much cheaper)
        model             the model id the provider says answered

    Anything the provider does not report is absent rather than zero. Zero and
    "not reported" are different facts and conflating them is how a cost report
    starts lying.
    """
    usage = _get(response, "usage")
    if usage is None:
        return None

    out = {}

    # Anthropic names first, then OpenAI's for the same quantity.
    tokens_in = _int(_get(usage, "input_tokens"))
    if tokens_in is None:
        tokens_in = _int(_get(usage, "prompt_tokens"))

    tokens_out = _int(_get(usage, "output_tokens"))
    if tokens_out is None:
        tokens_out = _int(_get(usage, "completion_tokens"))

    cache_write = _int(_get(usage, "cache_creation_input_tokens"))
    cache_read = _int(_get(usage, "cache_read_input_tokens"))

    # OpenAI reports cache hits nested, and does not report cache writes at all.
    if cache_read is None:
        cache_read = _int(_get(_get(usage, "prompt_tokens_details"),
                               "cached_tokens"))

    # OpenAI reports reasoning tokens nested. Anthropic folds thinking into
    # output_tokens and gives no separate figure, so this stays absent there.
    reasoning = _int(_get(_get(usage, "completion_tokens_details"),
                          "reasoning_tokens"))

    # OpenAI's prompt_tokens is inclusive of cached tokens; Anthropic's
    # input_tokens excludes them. Normalise to "billed at base input rate" so a
    # cold and a warm run are comparable across providers.
    if (tokens_in is not None and cache_read
            and _get(usage, "prompt_tokens") is not None
            and _get(usage, "input_tokens") is None):
        tokens_in = max(0, tokens_in - cache_read)

    for key, value in (("tokens_in", tokens_in),
                       ("tokens_out", tokens_out),
                       ("tokens_reasoning", reasoning),
                       ("cache_write", cache_write),
                       ("cache_read", cache_read)):
        if value is not None:
            out[key] = value

    model = _get(response, "model")
    if isinstance(model, str) and model:
        out["model"] = model

    return out or None
