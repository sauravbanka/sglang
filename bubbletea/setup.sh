#!/usr/bin/env bash
# setup.sh — bootstrap the BubbleTea *SGLang* environment (bubble profiler).
#
# Mirrors the vLLM BubbleTea setup (uv + .venv 3.12 + editable install with
# PREBUILT CUDA-kernel wheels — no kernel is compiled from source). Same cluster.
#
# Usage:
#   bash bubbletea/setup.sh
#
# Optional environment variables:
#   VENV_DIR    — venv directory name          (default: .venv)
#   MODEL_DIR   — path to DeepSeek-V2-Lite      (default: ~/models/deepseek-ai/DeepSeek-V2-Lite)
#   HF_TOKEN    — HuggingFace token (for model download if not cached)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> sglang/ repo root
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[ok]${NC}  $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[err]${NC}  $*" >&2; exit 1; }

echo "=== BubbleTea SGLang setup ==="
echo "Repo: $REPO_ROOT"
echo

# ── 1. Prerequisites ──────────────────────────────────────────────────────────
echo "--- Checking prerequisites ---"

# CUDA (kernels come as prebuilt cu13 wheels, but torch/driver must be present)
if ! command -v nvcc &>/dev/null && ! nvidia-smi &>/dev/null; then
    die "CUDA not found. This project requires 2× A100 or 2× H100 GPUs."
fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ok "Found $GPU_COUNT GPU(s): $GPU_NAME"
[ "$GPU_COUNT" -lt 2 ] && warn "Bubble profiling needs 2 GPUs (TP=2 EP=2). $GPU_COUNT found."

# uv (bootstrap if missing — same as the vLLM setup)
if ! command -v uv &>/dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
ok "uv: $(uv --version)"

# Rust toolchain — required ONLY to build the small setuptools-rust router
# extension (pyproject.toml:[tool.setuptools-rust]). This is NOT a CUDA build;
# the CUDA kernels below are prebuilt wheels. Bootstrap like uv if absent.
if ! command -v cargo &>/dev/null; then
    warn "Rust toolchain not found (needed for the SGLang router extension)."
    echo "Installing rustup ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
fi
ok "cargo: $(cargo --version 2>/dev/null || echo 'unavailable')"

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo
echo "--- Setting up Python environment ---"
VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating $VENV_DIR (Python 3.12) ..."
    uv venv "$VENV_DIR" --python 3.12
    ok "Created $VENV_DIR"
else
    ok "$VENV_DIR already exists — skipping creation"
fi

# ── 3. Install SGLang (editable fork) + PREBUILT kernel wheels ────────────────
echo
echo "--- Installing SGLang (editable fork, prebuilt CUDA-kernel wheels) ---"
echo "Kernels (flashinfer, flash-attn-4, cutlass-dsl, sgl-kernel) install as"
echo "prebuilt cu13 wheels — nothing CUDA is compiled here. First run is slow."
echo
# The package lives under python/. --torch-backend=auto picks the matching torch
# wheel for the cluster's CUDA, exactly like the vLLM setup's install line.
uv pip install --python "$VENV_DIR/bin/python" -e "python[all]" --torch-backend=auto
ok "SGLang (fork) installed with prebuilt kernels"

# BubbleTea extra deps (parity with the vLLM setup; datasets/transformers are
# already pulled by sglang, listed explicitly so the two envs match).
echo "Installing BubbleTea dependencies ..."
uv pip install --python "$VENV_DIR/bin/python" \
    "safetensors>=0.4" "datasets>=2.18" "transformers>=4.40" "requests>=2.31"
ok "BubbleTea dependencies installed"

# ── 4. DeepEP check (needed for the Waterfill / dispatch-combine bubble path) ─
echo
echo "--- DeepEP (for --moe-a2a-backend deepep + Waterfill) ---"
if "$VENV_DIR/bin/python" -c "import deep_ep" 2>/dev/null; then
    ok "deep_ep importable"
else
    warn "deep_ep not importable. The bubble profiler measures the DeepEP"
    echo "  dispatch/combine window and Waterfill REQUIRES --moe-a2a-backend deepep."
    echo "  Per your 'prebuilt only' constraint, this script does NOT build DeepEP"
    echo "  from source. Use the cluster's prebuilt DeepEP (module load / the SGLang"
    echo "  CUDA image that ships deep_ep), then re-run. Baseline (non-DeepEP) bubble"
    echo "  timing still works without it, but Waterfill runs do not."
fi

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo
echo "=== Setup complete ==="
MODEL_DIR="${MODEL_DIR:-$HOME/models/deepseek-ai/DeepSeek-V2-Lite}"
if [ -d "$MODEL_DIR" ]; then
    ok "DeepSeek-V2-Lite found at $MODEL_DIR"
else
    warn "Model not found at $MODEL_DIR — download with:"
    echo "    export HF_TOKEN=<token>"
    echo "    $VENV_DIR/bin/python -c \"from huggingface_hub import snapshot_download; \\"
    echo "      snapshot_download('deepseek-ai/DeepSeek-V2-Lite', local_dir='$MODEL_DIR')\""
fi
echo
echo "Activate: source $VENV_DIR/bin/activate"
echo
echo "Profile bubbles (baseline, no balancer):"
echo "  SGLANG_BUBBLE_PROFILE=1 $VENV_DIR/bin/python -m sglang.launch_server \\"
echo "    --model-path $MODEL_DIR --trust-remote-code \\"
echo "    --tp-size 2 --ep-size 2 --moe-a2a-backend deepep --disable-cuda-graph --port 30000"
echo "  # then: python bubbletea/analyze_bubbles.py drive --tag no_waterfill --out ./bubble_out"
echo
echo "See bubbletea/README.md for the full Waterfill on/off procedure."
