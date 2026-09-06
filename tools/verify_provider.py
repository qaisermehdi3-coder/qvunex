"""Check Qvunex against a real provider instead of against our own assumptions.

Every test in this repo uses fake response objects we wrote. They prove the code
does what we *believe* providers do. They do not prove providers do that. If a
field name is wrong, or a count means something other than what we assumed, all
of those tests still pass and the number is still wrong.

This script closes that gap. It makes real calls, prints exactly what came back,
and checks four things that can only be answered by a provider, not by us:

  1. Does our extractor find the token counts at all?
  2. Is the reported input count inclusive of cached tokens, or on top of them?
  3. Are thinking tokens counted inside output tokens, or added to them?
  4. Does a repeated prompt actually produce a cache read?

(2) and (3) matter more than they look. Each one is worth roughly a third of the
bill on the workloads where it applies, and getting either backwards produces a
confident wrong number rather than an obvious error.

Run it with a free Google AI Studio key -- no card, no charge:

    pip install google-genai
    python verify_provider.py            # it will ask for the key

It costs nothing on the free tier and prints a verdict you can act on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

try:
    from qvunex.usage import extract
except Exception:                                    # running the file alone
    extract = None

MODEL = os.environ.get("QVUNEX_VERIFY_MODEL", "gemini-2.5-flash")

# Long enough to clear the implicit-caching minimum. Cache minimums are in the
# low thousands of tokens; below them the cache is silently skipped, which is
# the same trap taskcost.py warns about on the Anthropic side.
FILLER = ("The following is reference material for a question answering task. "
          "It is deliberately repetitive so that the same prefix can be sent "
          "twice and produce a cache read on the second call. ") * 220

QUESTION = "In one short sentence, what is this reference material for?"


def _u(resp):
    return getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)


def _g(u, *names):
    for n in names:
        v = getattr(u, n, None)
        if v is None and isinstance(u, dict):
            v = u.get(n)
        if v is not None:
            return int(v)
    return None


def show(label, resp):
    u = _u(resp)
    if u is None:
        print(f"  {label}: no usage metadata on the response at all")
        return None
    fields = {
        "prompt": _g(u, "prompt_token_count", "promptTokenCount"),
        "cached": _g(u, "cached_content_token_count", "cachedContentTokenCount"),
        "output": _g(u, "candidates_token_count", "candidatesTokenCount"),
        "thinking": _g(u, "thoughts_token_count", "thoughtsTokenCount"),
        "total": _g(u, "total_token_count", "totalTokenCount"),
    }
    print(f"\n  {label}")
    print("    provider reported:")
    for k, v in fields.items():
        print(f"      {k:<10} {'-' if v is None else format(v, ',')}")
    if extract is not None:
        print(f"    qvunex extracted: {extract(resp)}")
    return fields


def verdict(name, passed, detail):
    mark = "PASS" if passed else "FAIL" if passed is False else "??  "
    print(f"  [{mark}] {name}")
    if detail:
        print(f"         {detail}")


def main():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = input("Paste your Google AI Studio API key: ").strip()
        except EOFError:
            key = ""
    if not key:
        print("No key given. Get a free one at aistudio.google.com/apikey")
        return 1
    os.environ["GOOGLE_API_KEY"] = key

    from google import genai
    from google.genai import types

    client = genai.Client()
    print(f"model: {MODEL}\n" + "=" * 66)

    # --- 1. a plain call, thinking off where the model allows it -----------
    try:
        plain = client.models.generate_content(
            model=MODEL, contents="Say the word: ok",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)))
        a = show("plain call, thinking disabled", plain)
    except Exception as e:
        print(f"  plain call failed: {type(e).__name__}: {e}")
        a = None

    # --- 2. the same long prompt twice, to try for a cache read ------------
    b = c = None
    try:
        long_prompt = FILLER + "\n\n" + QUESTION
        cold = client.models.generate_content(model=MODEL, contents=long_prompt)
        b = show("long prompt, first time (expect a cold run)", cold)
        warm = client.models.generate_content(model=MODEL, contents=long_prompt)
        c = show("same prompt again (expect a cache read)", warm)
    except Exception as e:
        print(f"  cache calls failed: {type(e).__name__}: {e}")

    # --- 3. a thinking call ------------------------------------------------
    d = None
    try:
        think = client.models.generate_content(
            model=MODEL,
            contents="Work out 17 * 24 step by step, then give the answer.")
        d = show("thinking call", think)
    except Exception as e:
        print(f"  thinking call failed: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("  WHAT THIS TELLS US")
    print("=" * 66)

    verdict("our extractor reads this provider at all",
            bool(extract and a and extract(plain)),
            "" if extract else "run from the repo so qvunex is importable")

    # The question we cannot answer from documentation: are thinking tokens
    # inside the output count or added to it? The provider's own total settles
    # it -- no interpretation needed.
    sample = d if (d and d.get("thinking")) else None
    if sample and sample.get("total") is not None:
        base = (sample.get("prompt") or 0) + (sample.get("output") or 0)
        with_think = base + (sample.get("thinking") or 0)
        if sample["total"] == with_think and sample["total"] != base:
            verdict("thinking tokens are billed ON TOP of output", True,
                    f"total {sample['total']:,} = prompt + output + thinking. "
                    "Our pricing must ADD them here. It currently does not, "
                    "which undercounts thinking-heavy work.")
        elif sample["total"] == base:
            verdict("thinking tokens are counted INSIDE output", True,
                    f"total {sample['total']:,} = prompt + output. Adding them "
                    "again would inflate the bill. Our pricing is correct.")
        else:
            verdict("thinking token accounting", None,
                    f"total {sample['total']:,} matches neither "
                    f"{base:,} nor {with_think:,}. Something else is in there.")
    else:
        verdict("thinking token accounting", None,
                "no thinking tokens came back, so this is untested")

    if c and c.get("cached"):
        cold_p = (b or {}).get("prompt")
        warm_p = c.get("prompt")
        verdict("a repeated prompt produced a cache read", True,
                f"{c['cached']:,} cached tokens on the second call")
        if cold_p and warm_p:
            if warm_p >= c["cached"]:
                verdict("reported input INCLUDES cached tokens", True,
                        f"prompt {warm_p:,} >= cached {c['cached']:,}, so we "
                        "must subtract to get tokens billed at the base rate. "
                        "That is what the extractor does.")
            else:
                verdict("reported input EXCLUDES cached tokens", True,
                        f"prompt {warm_p:,} < cached {c['cached']:,}. Do NOT "
                        "subtract -- the extractor is wrong for this provider.")
    else:
        verdict("cache read on a repeated prompt", None,
                "no cached tokens reported. Either the prompt was under the "
                "cache minimum or implicit caching did not fire. Not a bug in "
                "our code, but it means the cache path is still unverified.")

    print("\n  Nothing here was inferred from documentation. Every line above")
    print("  came from what the provider actually returned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
