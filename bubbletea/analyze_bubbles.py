#!/usr/bin/env python3
"""Drive an SGLang server and analyze EP-imbalance bubbles.

Works with the ``SGLANG_BUBBLE_PROFILE=1`` profiler added to the MoE
dispatch/combine path (see ``sglang/srt/layers/moe/bubble_profile.py``).

Two deliverables:

  (1) How much of the bubble does Waterfill (shared-expert rebalance) remove?
      -> compare the cross-rank spread of ``bubble_ms`` with Waterfill off vs on.

  (2) Is Waterfill's ``rank_load`` a good free bubble predictor?
      -> correlate the predicted per-rank load spread against the measured bubble.

Usage
-----
Launch a server with the profiler on and cuda graphs OFF (events do not time
inside captured graphs), then:

    # baseline server (no --enable-deepep-waterfill):
    python analyze_bubbles.py drive --url http://127.0.0.1:30000 \
        --tag no_waterfill --out ./bubble_out --lengths 4096,8192,16384,32768

    # restart server WITH --enable-deepep-waterfill, then:
    python analyze_bubbles.py drive --url http://127.0.0.1:30000 \
        --tag waterfill --out ./bubble_out --lengths 4096,8192,16384,32768

    python analyze_bubbles.py analyze --out ./bubble_out \
        --base no_waterfill --wf waterfill

``drive`` snapshots the server's ``/tmp/sglang_bubble_rank*.jsonl`` into
``{out}/{tag}_rank{r}.jsonl`` (run it on the same node as the server).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict

import requests

TMP_GLOB = "/tmp/sglang_bubble_rank*.jsonl"
REPEAT_TOKEN = 100  # single repeated token id => concentrated routing => max imbalance


# --------------------------------------------------------------------------- #
# drive
# --------------------------------------------------------------------------- #
def _generate(url: str, input_ids: list[int], max_new_tokens: int = 1) -> None:
    r = requests.post(
        f"{url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        },
        timeout=600,
    )
    r.raise_for_status()


def drive(args) -> None:
    url = args.url.rstrip("/")
    lengths = [int(x) for x in args.lengths.split(",")]
    os.makedirs(args.out, exist_ok=True)

    # Warmup (not measured; also primes weights / DeepEP buffers).
    for _ in range(args.warmup):
        _generate(url, [REPEAT_TOKEN] * min(lengths))

    for L in lengths:
        prompt = [REPEAT_TOKEN] * L
        for i in range(args.repeats):
            _generate(url, prompt)
            print(f"  drove length={L} rep={i + 1}/{args.repeats}", flush=True)
    # One extra request so the profiler's layer-id-repeat rule flushes the LAST
    # measured forward pass (each pass is resolved when the next one begins).
    _generate(url, [REPEAT_TOKEN] * min(lengths))
    time.sleep(1.0)

    snapped = 0
    for path in sorted(glob.glob(TMP_GLOB)):
        rank = path.split("rank")[-1].split(".")[0]
        dst = os.path.join(args.out, f"{args.tag}_rank{rank}.jsonl")
        shutil.copyfile(path, dst)
        n = sum(1 for _ in open(dst))
        print(f"snapshot {path} -> {dst} ({n} records)")
        snapped += 1
    if snapped == 0:
        sys.exit(
            f"No {TMP_GLOB} found. Is SGLANG_BUBBLE_PROFILE=1 set, ep_size>1, and "
            "cuda graph disabled? (run this on the same node as the server)"
        )


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _load(out_dir: str, tag: str) -> dict[int, list[dict]]:
    """Return {rank: [records]} for a config tag."""
    per_rank: dict[int, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(out_dir, f"{tag}_rank*.jsonl"))):
        rank = int(path.split("rank")[-1].split(".")[0])
        per_rank[rank] = [json.loads(line) for line in open(path)]
    if not per_rank:
        sys.exit(f"No records for tag '{tag}' in {out_dir}")
    return per_rank


def _median_bubble_by_len_rank(per_rank: dict[int, list[dict]]):
    """{n_tokens: {rank: median bubble_ms}} aggregated over layers+forwards."""
    acc: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rank, recs in per_rank.items():
        for r in recs:
            acc[r["n_tokens"]][rank].append(r["bubble_ms"])
    return {
        n: {rk: statistics.median(v) for rk, v in ranks.items()}
        for n, ranks in acc.items()
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx**0.5 * vy**0.5)


def analyze(args) -> None:
    base = _load(args.out, args.base)
    wf = _load(args.out, args.wf)

    # ---- Deliverable 1: % of bubble removed by Waterfill --------------------
    b_base = _median_bubble_by_len_rank(base)
    b_wf = _median_bubble_by_len_rank(wf)
    print("\n=== (1) Bubble reduction by Waterfill (shared-expert rebalance) ===")
    print(
        f"{'n_tokens':>9} | {'base spread':>12} | {'wf spread':>12} | "
        f"{'reduced':>8} | base per-rank bubble_ms"
    )
    print("-" * 88)
    for n in sorted(set(b_base) & set(b_wf)):
        base_ranks = b_base[n]
        wf_ranks = b_wf[n]
        base_spread = max(base_ranks.values()) - min(base_ranks.values())
        wf_spread = max(wf_ranks.values()) - min(wf_ranks.values())
        pct = 100.0 * (base_spread - wf_spread) / base_spread if base_spread else 0.0
        per_rank_str = ", ".join(
            f"r{rk}={ms:.2f}" for rk, ms in sorted(base_ranks.items())
        )
        print(
            f"{n:>9} | {base_spread:>10.2f}ms | {wf_spread:>10.2f}ms | "
            f"{pct:>6.1f}% | {per_rank_str}"
        )

    # ---- Deliverable 2: rank_load estimate vs measured bubble ---------------
    print("\n=== (2) Waterfill rank_load estimate vs measured bubble ===")
    xs, ys = [], []  # predicted imbalance (rank_load spread), measured bubble_ms
    for recs in wf.values():
        for r in recs:
            if r.get("rank_load_spread") is not None:
                xs.append(float(r["rank_load_spread"]))
                ys.append(float(r["bubble_ms"]))
    if not xs:
        print("  no rank_load recorded (was --enable-deepep-waterfill on?)")
        return
    r = _pearson(xs, ys)
    print(f"  paired samples : {len(xs)}")
    if r is None:
        print("  correlation    : undefined (too few / zero-variance samples)")
    else:
        print(f"  Pearson r      : {r:+.3f}")
        print(f"  R^2            : {r * r:.3f}")
        verdict = (
            "strong -> rank_load is a usable free bubble predictor"
            if r * r >= 0.6
            else "weak -> a dedicated timing predictor is still needed"
        )
        print(f"  verdict        : {verdict}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("drive", help="send repeated-token prompts, snapshot JSONL")
    d.add_argument("--url", default="http://127.0.0.1:30000")
    d.add_argument("--tag", required=True, help="config name, e.g. no_waterfill / waterfill")
    d.add_argument("--out", default="./bubble_out")
    d.add_argument("--lengths", default="4096,8192,16384,32768")
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--warmup", type=int, default=2)
    d.set_defaults(func=drive)

    a = sub.add_parser("analyze", help="compute reduction % and estimator correlation")
    a.add_argument("--out", default="./bubble_out")
    a.add_argument("--base", default="no_waterfill")
    a.add_argument("--wf", default="waterfill")
    a.set_defaults(func=analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
