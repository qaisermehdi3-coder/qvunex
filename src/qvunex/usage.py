"""Read token counts off a provider's response object.

Every provider reports usage in a slightly different shape, and the differences
are not cosmetic. Two of them matter enough to be the reason this file exists:

**Cache tokens are not a discount, they are separate line items.** A cache write
costs *more* than not caching at all (1.25x input price on a 5-minute TTL, 2x on
an hour); a cache read costs 0.1x. Folding them into one "input tokens" number
makes a cold run and a warm run look identical when they differ by an order of
magnitude.

**Reasoning tokens are billed, rarely broken out, and counted two different
ways.** On reasoning-heavy work they routinely exceed the visible output. Where a
provider reports them we record them separately -- but recording them is only
half the job, because providers disagree about whether they are already inside
the output count:

    OpenAI     reasoning_tokens is a SUBSET of completion_tokens. Adding it
               again inflates the bill.
    Google     thoughts_token_count is ADDITIVE. Their own total_token_count
               equals prompt + candidates + thoughts (measured, see _google).
               Not adding it undercounts the bill.
    Anthropic  thinking is folded into output_tokens and never reported apart,
               so there is nothing to add and nothing to double count.

Two conventions, opposite directions, and no error is raised either way. So each
usage dict carries `reasoning_billed` saying which one applies, and pricing acts
on that rather than on a global assumption.

No dependencies and no imports of any provider SDK. Everything here is
duck-typed, so a new provider that reports the same field names works without a
change, and one that doesn't returns None instead of raising.
"""


# Which convention the provider uses for reasoning tokens. Recorded per call,
# because it differs by provider and getting it wrong is worth a third to four
# fifths of the bill on thinking-heavy work.
INCLUDED = "included"   # already inside tokens_out; do not add again
EXTRA = "extra"         # billed on top of tokens_out; must be added


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


def _first(obj, *names):
    for n in names:
        v = _int(_get(obj, n))
        if v is not None:
            return v
    return None


def _reconcile(out):
    """Check our own arithmetic against the provider's own total.

    Where a provider reports a total, it is the one number we did not derive.
    Adding up what we recorded and comparing is therefore a free audit of the
    whole extraction: a renamed field, a count that turned out to be nested
    inside another, a new kind of token nobody has told us about -- each of them
    shows up here as a difference, where otherwise they would show up as a
    slightly wrong bill and nothing else.

    Double counting reasoning tokens is the specific failure this catches. Both
    directions of that mistake are silent at the provider, and both are visible
    here immediately.

    A difference is recorded, never corrected. We do not know which side is
    wrong, and quietly adjusting a number to make a check pass is how a meter
    starts lying.
    """
    total = out.get("tokens_total_reported")
    if total is None:
        return
    accounted = (out.get("tokens_in", 0) + out.get("cache_read", 0)
                 + out.get("cache_write", 0) + out.get("tokens_out", 0))
    if out.get("reasoning_billed") == EXTRA:
        accounted += out.get("tokens_reasoning", 0)
    diff = total - accounted
    if diff:
        out["tokens_unaccounted"] = diff


def _google(usage, response=None):
    """Google's shape. Same facts, different names, one open question.

        prompt_token_count          input, believed inclusive of cached
        cached_content_token_count  tokens served from context cache
        candidates_token_count      output
        thoughts_token_count        thinking
        total_token_count           the provider's own sum

    Thinking tokens here are **additive**, not a breakdown. Measured against
    gemini-3.6-flash on 2026-09-06, three calls, every one exact:

        17    + 144 + 589 =   750  (reported total)
        4,632 +  26 + 236 = 4,894
        4,632 +  32 + 238 = 4,902

    total_token_count = prompt + candidates + thoughts. Google's own arithmetic,
    not our reading of anyone's documentation. So thoughts are NOT inside
    candidates_token_count, and they are billed at the output rate on top of it.

    This is the opposite of OpenAI, where reasoning_tokens is a subset of
    completion_tokens and adding it again inflates the bill. Same concept, two
    conventions, no error raised either way. On the first call above thinking
    was 589 of 750 tokens -- 79% of the call -- so getting this backwards is not
    a rounding difference, it is most of the number.

    We record which convention applies rather than assuming one globally.
    """
    out = {}
    prompt = _first(usage, "prompt_token_count", "promptTokenCount")
    cached = _first(usage, "cached_content_token_count", "cachedContentTokenCount")
    candidates = _first(usage, "candidates_token_count", "candidatesTokenCount")
    thoughts = _first(usage, "thoughts_token_count", "thoughtsTokenCount")
    total = _first(usage, "total_token_count", "totalTokenCount")

    # Normalise to "billed at the base input rate", as for OpenAI.
    if prompt is not None and cached:
        prompt = max(0, prompt - cached)

    for key, value in (("tokens_in", prompt),
                       ("tokens_out", candidates),
                       ("tokens_reasoning", thoughts),
                       ("cache_read", cached),
                       ("tokens_total_reported", total)):
        if value is not None:
            out[key] = value
    if thoughts:
        out["reasoning_billed"] = EXTRA

    model = _get(response, "model_version") or _get(response, "modelVersion")
    if isinstance(model, str) and model:
        out["model"] = model
    _reconcile(out)
    return out or None


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
        # Google calls it usage_metadata (usageMetadata over REST) and names
        # every field differently. Same facts, different spelling.
        usage = _get(response, "usage_metadata") or _get(response, "usageMetadata")
        if usage is not None:
            return _google(usage, response)
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
    if reasoning:
        # OpenAI nests reasoning inside completion_tokens_details, and those
        # tokens are already counted in completion_tokens.
        out["reasoning_billed"] = INCLUDED

    total = _int(_get(usage, "total_tokens"))
    if total is not None:
        out["tokens_total_reported"] = total

    model = _get(response, "model")
    if isinstance(model, str) and model:
        out["model"] = model

    _reconcile(out)
    return out or None
