# BubbleTea — SGLang EP-bubble profiler

Measures the EP-imbalance **bubble** in SGLang's MoE dispatch/combine path, and
answers two questions about Waterfill (shared-expert rebalance):

1. **What % of the bubble does Waterfill remove?**
2. **Is Waterfill's `rank_load` a good enough free predictor to skip a separate
   bubble timer?**

## What was added

| File | Change |
|------|--------|
| `srt/environ.py` | `SGLANG_BUBBLE_PROFILE` env flag |
| `srt/layers/moe/bubble_profile.py` | the profiler (CUDA events + deferred JSONL dump) |
| `srt/layers/moe/fused_moe_triton/layer.py` | events around `dispatch` / GEMM / `combine` in `forward_impl` |
| `srt/layers/moe/deepep_waterfill.py` | stash predicted `rank_load` per layer |

The profiler brackets `forward_impl` with four CUDA events
(`e0 →dispatch→ e1 →GEMM→ e2 →combine→ e3`). The **`e2→e3` (combine) window is the
bubble**: the lightly-loaded rank cannot leave `combine()` until the heavy rank
finishes and the DeepEP all-to-all completes. Events are resolved with a single
`torch.cuda.synchronize()` at each forward-pass boundary (detected by layer-id
repeat) — never per layer, which would serialize the async collective. Output is
one JSON-lines file per EP rank: `/tmp/sglang_bubble_rank{rank}.jsonl`.

## Constraints (important)

- **Disable cuda graphs** (`--disable-cuda-graph`). In this fork *both* prefill
  and decode are cuda-graphed by default, and events captured into a graph record
  at capture time, not replay time. The profiler self-skips capture, so without
  this flag you simply get no records.
- Requires `ep_size > 1` (the profiler self-guards on `moe_ep_size > 1`).
- Profiles the **prefill** forward (large prompts) — that's where the bubble is
  biggest (it scales with context length).

## Run (DeepSeek-V2-Lite, 2×A100)

DeepSeek-V2-Lite has 2 shared experts, so Waterfill applies and it fits 2×A100.

**Baseline (no balancer):**
```bash
SGLANG_BUBBLE_PROFILE=1 python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2-Lite \
    --trust-remote-code \
    --tp-size 2 --ep-size 2 \
    --moe-a2a-backend deepep \
    --disable-cuda-graph \
    --port 30000
```
```bash
python bubbletea/analyze_bubbles.py drive \
    --url http://127.0.0.1:30000 --tag no_waterfill --out ./bubble_out \
    --lengths 4096,8192,16384,32768
```

**Waterfill on** — kill the server, relaunch with `--enable-deepep-waterfill`
added (everything else identical), then:
```bash
python bubbletea/analyze_bubbles.py drive \
    --url http://127.0.0.1:30000 --tag waterfill --out ./bubble_out \
    --lengths 4096,8192,16384,32768
```

**Analyze:**
```bash
python bubbletea/analyze_bubbles.py analyze \
    --out ./bubble_out --base no_waterfill --wf waterfill
```

Prints (1) per-length cross-rank bubble spread base-vs-Waterfill with % reduced,
and (2) the Pearson r / R² of predicted `rank_load` spread against measured
`bubble_ms`.

## Notes

- Run `drive` on the **same node** as the server (it snapshots `/tmp/…jsonl`).
- Each server process truncates its JSONL on first write, so use a **fresh
  server** per config (which you do anyway — Waterfill is a launch flag).
- To first reproduce vLLM↔SGLang bubble parity, run only the baseline config on
  the same model in both engines and compare `bubble_ms` before adding Waterfill.
- LPLB (`--ep-dispatch-algorithm lp`) is intentionally not swept: at ep_size=2 it
  has no replicas to redistribute across, so it's a no-op at this scale.
