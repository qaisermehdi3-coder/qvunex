# Agent cost audit

You run a small file on your own machine for a day. You send me the file it
writes. I send you back where your LLM money actually goes.

This is for teams running agents in production who can see the monthly bill
but cannot say which task or step is behind it, or why it moved.

---

## What you get

A written note from your own traffic, covering:

- **Cost per finished task** — the average and the p95, and which tasks sit in
  that p95. The average hides the tasks that actually hurt.
- **Which step costs the most** inside each task, where you have named steps.
- **How much spend nobody owns** — calls that happened inside a task but that
  nothing labelled. Shown on their own line, never folded into a total.
- **Calls whose cost is unknown** — the provider sent back no usage at all.
  Most dashboards show these as free.
- **Failed calls**, and tasks that crashed part way and still cost money.
- **Cache written but never read** — by Anthropic's published pricing a cache
  write costs more than not caching at all, so this is pure surcharge.
- **Thinking tokens against visible output**, on Gemini and OpenAI, which
  report them separately. On Gemini, a call whose answer was 3 tokens cost 138
  output tokens once the thinking was counted. Anthropic folds thinking into
  output and does not report it apart, so on Anthropic this line is not possible.
- **Where the provider's own total and the parts do not add up.**
- **Calls where a gateway dropped the cache counts.** On Anthropic, cached input
  is left out of `input_tokens`, so a proxy that drops the cache fields makes
  most of the input disappear from your cost numbers. One public report showed a
  call costing about $0.022 recorded as $0.000012. These calls are flagged, never
  read as zero. Checked against the shapes in that report, not yet a live gateway.
- **Running your own model server (vLLM)?** The note also covers what one
  request costs on your GPUs at the load you actually ran, idle time included,
  and the calls your client never got a count for, reconciled against the
  server's own counters. Tested live on vLLM 0.27.1: client and server totals matched
  exactly, and the one call that came back with no usage was
  recovered exactly from the server side.

And three to five specific changes, each tied to a number in your data.

## What you do

1. Put one file in your project. No install, standard library only:
   <https://github.com/qaisermehdi3-coder/qvunex/blob/main/qvunex_single.py>
2. Wrap your client. Name your tasks, and your steps where you know them:

   ```python
   from qvunex_single import wrap, task, step
   client = wrap(your_client)

   with task("support reply"):
       with step("draft"):
           client.messages.create(...)
   ```

3. Run your normal traffic for a day. A week if a day is quiet.
4. Send me the file it writes, `~/.qvunex/events.jsonl`.

## What I see, and what I don't

Tested, not assumed. I made a call with a customer email in the prompt, an API
key on the client, a private reply, a user id and a response id, then read the
file back. None of them were in it. This is that call's complete record, nothing
removed:

```
{"type": "call", "schema": "0.3", "t": 1790184319.620305, "model": "claude-x", "task": "support reply", "task_id": "ff40a33a5f77", "owner": "draft", "seconds": 1.8e-05, "status": "ok", "attempt": 1, "tokens_in": 120, "tokens_out": 40, "cache_write": 0, "cache_read": 0, "tokens_reasoning": 0, "reasoning_billed": "included", "tokens_total_reported": 160}
```

The only words from you in it are your model names and the names you gave your
tasks and steps. It is plain text, one line per call — read every line before
you send it.

Or check it yourself: `python3 qvunex_single.py --selftest` repeats that exact
test on your own machine, along with the attribution and cost behaviour
described above.

## What I can't tell you

- **If your framework does not say which subagent made a call, nothing can
  invent it.** Those calls show up as unowned. That is a finding, not a failure:
  it tells you how much of your bill you currently cannot attribute.
- **Calls made on worker threads** land outside any task on Python 3.10 to 3.13
  unless you pass the task along. The file explains how.
- **Prices** are published list prices unless you tell me your negotiated rates.
- **OpenAI chat streams** carry no usage unless you pass
  `stream_options={"include_usage": True}`. Without it, those calls show up as
  usage missing and the note says how many. They are never counted as free.
- **Retries inside your provider's SDK are invisible** to anything that wraps
  the client, this included. In a live test Google's SDK quietly retried a
  failing call for 25 seconds, and it shows as one failed call, with the time of
  all the tries together.
- **Provider coverage, honestly:** Gemini is tested live, for plain, streamed
  and async calls and for calls that fail. A streamed call that also thinks has
  not been caught live yet. Anthropic, OpenAI (Chat Completions and the
  Responses API) and LangChain's usage are handled from each one's documented
  usage shape and checked by `--selftest`, but not yet against a live account.
  If you are on one of those, you would be the first, and I will check your
  first report line by line against your provider's own dashboard before
  telling you anything.

## Price

**$1,200. You pay after you have read the findings, not before.**

**If the note does not show you anything worth acting on, you pay nothing.**

**After you make the changes, run it again for a day and I will show you the
before and after, in dollars.** No extra charge.

It is for teams spending at least $3,000 a month on LLM calls. Below that, an
audit costs more than it is likely to find, and I would rather tell you that
than take the money.

## Start

Email **qvunexaudit@gmail.com**. Tell me which provider and framework you use,
or which GPUs and serving stack if you run your own models, and roughly how many
calls a day. I will tell you straight if it is not a fit.
