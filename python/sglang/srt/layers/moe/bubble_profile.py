"""EP-imbalance "bubble" profiler for the MoE dispatch/combine path.

Enabled via ``SGLANG_BUBBLE_PROFILE=1``. For each MoE-layer forward it records
CUDA-event timings for three phases of ``FusedMoE.forward_impl``:

    e0 --dispatch--> e1 --local expert GEMM--> e2 --combine--> e3

The ``combine`` window (e2 -> e3) contains the EP-imbalance *bubble*: on the
lightly-loaded rank, ``combine()`` cannot return until the heavy rank has
finished its experts and the DeepEP all-to-all completes, so the light rank
sits idle. Comparing ``bubble_ms`` across ranks (max - min) gives the harvestable
idle window BubbleTea targets.

Design constraints (mirrors the vLLM bubble profiler):

* The DeepEP dispatch/combine wait is a *stream-ordered GPU wait*, not a CPU
  block, so timing events straddling ``combine()`` observe the bubble without a
  per-layer ``synchronize()``. We therefore accumulate raw CUDA events across a
  whole forward pass and resolve them with a SINGLE ``torch.cuda.synchronize()``
  at the pass boundary. A per-layer sync would serialize the async collective
  and inflate the measured bubble.
* Pass boundaries are detected by *layer-id repeat*: when a ``layer_id`` we have
  already seen in this pass shows up again, a new forward pass has begun, so we
  flush (resolve + dump) the previous pass first.
* CUDA-graph capture is skipped (``is_current_stream_capturing()``): events
  captured into a graph record at capture time, not replay time. Profiling runs
  must therefore disable cuda graphs (``--disable-cuda-graph``).

When Waterfill is active, ``stash_rank_load`` records the balancer's predicted
per-rank load for the same ``layer_id`` (a free, pre-dispatch estimate). It is
emitted next to the measured ``bubble_ms`` so the estimate can be correlated
against ground truth (i.e. tested as a cheap bubble predictor).

Output: one JSON-lines file per EP rank, ``/tmp/sglang_bubble_rank{rank}.jsonl``.
Each line: ``layer_id, n_tokens, dispatch_ms, gemm_ms, bubble_ms, total_ms`` and,
when Waterfill is on, ``rank_load`` (per-rank vector) and ``rank_load_spread``.
"""

from __future__ import annotations

import atexit
import json
import threading
from typing import Dict, List, Optional

import torch

from sglang.srt.environ import envs

# Read once at import: the flag is set before the worker process starts, and this
# keeps ``should_record`` free of per-layer env lookups (zero overhead when off).
_ENABLED: bool = envs.SGLANG_BUBBLE_PROFILE.get()

_OUT_TEMPLATE = "/tmp/sglang_bubble_rank{rank}.jsonl"


def is_enabled() -> bool:
    return _ENABLED


def should_record(moe_ep_size: int) -> bool:
    """True only when profiling is on, EP is actually split, and we are not
    capturing a CUDA graph (graph-captured events do not time real replays)."""
    return (
        _ENABLED
        and moe_ep_size > 1
        and not torch.cuda.is_current_stream_capturing()
    )


def new_events() -> List[torch.cuda.Event]:
    """Four timing events: [before dispatch, after dispatch, after GEMM, after combine]."""
    return [torch.cuda.Event(enable_timing=True) for _ in range(4)]


class _BubbleProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: List[dict] = []  # records for the in-flight forward pass
        self._seen_layers: set = set()
        self._rank_load: Dict[int, torch.Tensor] = {}  # layer_id -> GPU clone
        self._rank: Optional[int] = None
        self._truncated = False

    def stash_rank_load(self, layer_id: int, rank_load: torch.Tensor) -> None:
        # Detached GPU clone; moved to host only during resolve (after the single
        # synchronize), so stashing stays async and never serializes the forward.
        self._rank_load[int(layer_id)] = rank_load.detach().clone()

    def submit(
        self, layer_id: int, n_tokens: int, ep_rank: int, events: List[torch.cuda.Event]
    ) -> None:
        with self._lock:
            if self._rank is None:
                self._rank = int(ep_rank)
            if layer_id in self._seen_layers:
                # A previously-seen layer id => a new forward pass has begun.
                self._resolve_locked()
            self._seen_layers.add(layer_id)
            self._pending.append(
                {"layer_id": int(layer_id), "n_tokens": int(n_tokens), "_ev": events}
            )

    def flush(self) -> None:
        with self._lock:
            self._resolve_locked()

    def _resolve_locked(self) -> None:
        if not self._pending:
            self._seen_layers.clear()
            self._rank_load.clear()
            return
        # Single sync for the whole pass, then read out all event deltas.
        torch.cuda.synchronize()
        rows = []
        for rec in self._pending:
            e0, e1, e2, e3 = rec.pop("_ev")
            rec["dispatch_ms"] = e0.elapsed_time(e1)
            rec["gemm_ms"] = e1.elapsed_time(e2)
            rec["bubble_ms"] = e2.elapsed_time(e3)
            rec["total_ms"] = e0.elapsed_time(e3)
            rl = self._rank_load.get(rec["layer_id"])
            if rl is not None:
                vals = rl.to("cpu").tolist()
                rec["rank_load"] = vals
                rec["rank_load_spread"] = (max(vals) - min(vals)) if vals else None
            rows.append(rec)
        self._write(rows)
        self._pending = []
        self._seen_layers.clear()
        self._rank_load.clear()

    def _write(self, rows: List[dict]) -> None:
        path = _OUT_TEMPLATE.format(rank=self._rank if self._rank is not None else 0)
        # Truncate on the first write of the process, append thereafter.
        mode = "a" if self._truncated else "w"
        self._truncated = True
        with open(path, mode) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


_PROFILER: Optional[_BubbleProfiler] = None


def _get() -> _BubbleProfiler:
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = _BubbleProfiler()
    return _PROFILER


def submit(
    layer_id: int, n_tokens: int, ep_rank: int, events: List[torch.cuda.Event]
) -> None:
    _get().submit(layer_id, n_tokens, ep_rank, events)


def stash_rank_load(layer_id: int, rank_load: torch.Tensor) -> None:
    if _ENABLED:
        _get().stash_rank_load(layer_id, rank_load)


def flush() -> None:
    """Resolve + dump the final (un-flushed) forward pass. Best-effort at exit."""
    if _PROFILER is not None:
        _PROFILER.flush()


# The layer-id-repeat rule leaves the LAST pass un-flushed until the next pass
# arrives; flush it on clean shutdown so single-shot runs still emit their tail.
atexit.register(flush)
