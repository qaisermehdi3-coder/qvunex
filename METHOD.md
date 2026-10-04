# How to price one GPU wake: billed against served

Version 0.3, 4 October 2026 (0.2 added rule 8; 0.3 adds SGLang). A short
method for working out what a self-hosted or scale-to-zero inference server
cost, from the bill and from the server's own counters. Anyone can use it; you
do not need qvunex to follow it. `qvunex_single.py --reconcile` and `--idle`
are one implementation.

Every rule below comes from a live run, listed at the end.

## 1. The two clocks

- **Billed time**: what the provider charges for, from the moment billing
  starts (machine or worker created) to the moment it stops (deleted). Busy or
  idle, it is all paid.
- **Served work**: what the inference server itself counted in that time:
  requests finished, prompt tokens, output tokens. For vLLM these are
  `vllm:request_success_total`, `vllm:request_prompt_tokens_sum` and
  `vllm:request_generation_tokens_sum` on `/metrics`. For SGLang (started
  with `--enable-metrics`) they are `sglang:num_requests_total`,
  `sglang:prompt_tokens_total` and `sglang:generation_tokens_total`; add up
  the streamed and not-streamed series.

Cost per request and cost per token are billed time priced, divided by served
work. Never the other way round, and never from a speed measured on one
request.

## 2. Split every wake into three parts

| part | from | to |
|---|---|---|
| boot | billing start | first counter snapshot (server answering) |
| serving window | first snapshot | second snapshot |
| tail | second snapshot | billing end |

The three add up to the billed time. Report each in seconds, as a share of the
bill, and in money.

## 3. Rules

1. **A part you did not measure is unknown, not zero.** If you do not know the
   tail, say so; the serving window is then unknown too, because it is what is
   left over.
2. **Paid with nothing served is its own line.** If a paid window finished 0
   requests, report it as such, with the amount, not as "unknown".
   Health checks and `/v1/models` calls do not count as served work: on vLLM
   0.27.1 and SGLang 0.5.21 they leave the request counters unchanged. On
   SGLang, `/health` even runs a one-token generation on the GPU and is still
   not counted, so GPU activity alone does not mean work was served.
3. **Price by whole-server output over the billed time, idle included.** With
   several requests at once, per-request speed falls while the server's total
   rises. Pricing from one request's speed overstates cost.
4. **Do not price a scale-to-zero server from the serving window alone.** Every
   wake is a cold start; pass the whole billed time.
5. **On a server that stays up, skip the first window after start**, or report
   it apart: it can run far slower. Take the first snapshot only once the
   server answers: SGLang counts its own startup warmup request.
6. **Counters belong to the server process.** Two servers sharing one GPU each
   count only their own work, so attribution per server holds without a GPU
   monitor.
7. **Check the meter against the server.** What your client recorded and what
   the server counted should match exactly; a gap is a finding, not noise.
8. **On a server that stays up, take snapshots on a fixed interval and judge
   each window on its own.** Every window between two snapshots is served,
   paid with nothing served, or unknown. A counter that is missing, or that
   went down (a restart resets it), makes that window unknown, never idle. Add
   up the three: the "nothing served" total is the part of the bill that
   bought no work. Name each snapshot by its Unix time so the window lengths
   survive copying (a file's own time can change when it is copied).

## 4. Evidence (vLLM 0.27.1 unless marked SGLang)

- Wake split, Colab T4, 2 October 2026, two machines: boot 70.9% / 70.2%,
  serving 2.6% / 2.6%, a 60 s health-check tail 26.6% / 27.1%.
- Published GKE run by a third party (vLLM on a T4, KEDA 300 s cooldown,
  600 s node timer), computed from its own timestamps: boot about 37%, serving
  about 1%, tail about 62%.
- Paid, nothing served: Colab T4, 29 September 2026, 60 s of `/v1/models` and
  `/health` calls, 0 requests counted.
- Two servers on one T4, 29 September 2026: 8 and 3 requests sent, each server
  counted only its own, both matching the client exactly.
- Per-request vs whole server, 8 requests at once: T4 122-124 vs 701-708 tok/s;
  H100 SXM 549-573 vs 3,194-3,348 tok/s. About 5.7x on both.
- Scale-to-zero boot share: T4 about 165 of 171 s; H100 SXM about 75 of 77 s.
- Windows over time, Colab T4, 3 October 2026: 6 snapshots 30 s apart, planned
  idle, busy, idle, idle, busy (busy = 4 chat requests, idle = one `/health`
  call). Marked exactly that way: 60 s served, 90 s (60%) paid with nothing
  served, 0 s unknown.
- SGLang 0.5.21, Colab T4, 4 October 2026: 8 chat requests counted as exactly
  the client's 8 / 264 / 80; 6 `/health` and 6 `/v1/models` calls counted as
  nothing; 2 streams without `stream_options` (no usage at the client) still
  counted; a 1 / 6 / 8 warmup request already counted at startup. With a
  wrapped client, meter and server matched on requests (6 and 6) and the token
  gap was exactly the two streams. Idle, busy, idle windows marked that way.

## 5. What this method does not cover

- Servers other than vLLM and SGLang, and servers that do not export request
  and token counters.
- Training jobs.
- Kubernetes GPU time-slicing and MIG, and comparison with DCGM, are not tested.

Corrections and counter-examples are welcome as issues on
https://github.com/qaisermehdi3-coder/qvunex.
