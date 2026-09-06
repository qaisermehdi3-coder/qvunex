import os
from google import genai
from google.genai import types

os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_KEY_HERE"
MODEL = "gemini-2.5-flash"
client = genai.Client()

def nums(r):
    u = r.usage_metadata
    return {
        "prompt":   getattr(u, "prompt_token_count", None),
        "cached":   getattr(u, "cached_content_token_count", None),
        "output":   getattr(u, "candidates_token_count", None),
        "thinking": getattr(u, "thoughts_token_count", None),
        "total":    getattr(u, "total_token_count", None),
    }

def show(label, r):
    f = nums(r)
    print(f"\n{label}")
    for k, v in f.items():
        print(f"   {k:<9} {'-' if v is None else format(v, ',')}")
    return f

# 1. a thinking call: does thinking add to output, or sit inside it?
t = show("THINKING CALL", client.models.generate_content(
    model=MODEL, contents="Work out 17 * 24 step by step, then answer."))

# 2. the same long prompt twice: does a cache read happen, and is the
#    reported input count inclusive of it?
filler = ("Reference material for a question answering task. "
          "Deliberately repetitive so the same prefix can be sent twice. ") * 220
p = filler + "\n\nIn one sentence, what is this for?"
cold = show("LONG PROMPT, FIRST TIME", client.models.generate_content(
    model=MODEL, contents=p))
warm = show("SAME PROMPT AGAIN", client.models.generate_content(
    model=MODEL, contents=p))

print("\n" + "=" * 56)
print("ANSWERS")
print("=" * 56)

if t["thinking"]:
    base = (t["prompt"] or 0) + (t["output"] or 0)
    plus = base + t["thinking"]
    if t["total"] == plus and t["total"] != base:
        print("thinking is billed ON TOP of output")
        print(f"  total {t['total']:,} = prompt + output + thinking")
        print("  -> qvunex must ADD thinking tokens. It currently does not.")
    elif t["total"] == base:
        print("thinking is counted INSIDE output")
        print(f"  total {t['total']:,} = prompt + output")
        print("  -> qvunex is correct not to add them.")
    else:
        print(f"unclear: total {t['total']:,}, "
              f"prompt+output {base:,}, plus thinking {plus:,}")
else:
    print("no thinking tokens came back - question still unanswered")

if warm["cached"]:
    print(f"\ncache read happened: {warm['cached']:,} cached tokens")
    if warm["prompt"] >= warm["cached"]:
        print(f"  prompt {warm['prompt']:,} INCLUDES the cached tokens")
        print("  -> qvunex must subtract. It does.")
    else:
        print(f"  prompt {warm['prompt']:,} EXCLUDES the cached tokens")
        print("  -> qvunex must NOT subtract. It does. That is a bug.")
else:
    print("\nno cache read - prompt may be under the cache minimum")
    print(f"  cold prompt was {cold['prompt']:,} tokens")
