# Try it on your own vLLM server (about 2 minutes to start)

You get: for every 5 minutes your GPU was up, whether it **served requests** or was
**paid for and served nothing**, in dollars at your hourly rate.

What it touches: it saves vLLM's own `/metrics` page to files every 5 minutes, and
later reads those files. The files hold counters and the model name, no prompts and
no replies (open one and check). `qvunex_single.py` has no network code and needs no
install. Nothing is sent anywhere.

## 1. Start saving snapshots (on the machine that can reach vLLM)

```bash
mkdir -p snaps
nohup sh -c 'while true; do t=$(date +%s); curl -sf localhost:8000/metrics -o snaps/.part && mv snaps/.part snaps/$t.txt; sleep 300; done' >/dev/null 2>&1 &
echo $! > snaps/loop.pid
```

Change `localhost:8000` if your server listens elsewhere. If vLLM runs in Kubernetes,
run `kubectl port-forward svc/<your-vllm-service> 8000:8000` first and leave it open
(we have not tested that path ourselves).

After 10 minutes, `ls snaps` should show files named like `1792262383.txt`. If the
server is down when the loop wakes, no file is written for that turn.

## 2. Leave it running

A few hours shows the pattern; a full day or a week shows the bill.

## 3. Read the result

```bash
curl -sLO https://raw.githubusercontent.com/qaisermehdi3-coder/qvunex/main/qvunex_single.py
python3 qvunex_single.py --idle snaps/*.txt --gpu-rate 1.10
```

Put your own price per hour after `--gpu-rate`. Stop the loop with
`kill $(cat snaps/loop.pid)`.

The output looks like this (from the test below):

```text
  1792262383.txt -> 1792262386.txt        3.0 s  PAID, NOTHING SERVED
  1792262386.txt -> 1792262389.txt        3.0 s  served 3 requests, 48 output tokens
  1792262389.txt -> 1792262392.txt        3.0 s  PAID, NOTHING SERVED
  1792262392.txt -> 1792262395.txt        3.0 s  PAID, NOTHING SERVED

  total                12.0 s
  served                3.0 s   25.0%  $0.000917
  nothing served        9.0 s   75.0%  $0.002750
  unknown               0.0 s    0.0%  $0.000000
```

A window is **unknown** when a counter is missing or the server restarted and its
counters reset. Unknown is never counted as idle.

If you want a second pair of eyes on the result, send the summary lines (or the
`snaps` folder) to qvunexaudit@gmail.com.

## How this page was tested

17 October 2026: the loop above, with `sleep 3` instead of 300, against a stand-in
server that replayed real `/metrics` text saved from vLLM 0.27.1 on a T4 (14 October,
in `benchmarks/data/selfhosted-t4-2026-10-14`): 5 files written, named by Unix time,
and the report above. With no server listening, curl fails and no file is written.
The `--idle` report itself was tested live against vLLM 0.27.1 on a Colab T4 on
3 October 2026 (see the README). Not tested by us: Kubernetes port-forward, SGLang
with this loop, and runs longer than a few minutes.
