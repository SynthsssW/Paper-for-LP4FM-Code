#!/usr/bin/env bash
# Minimal A40 pod setup. DESIGN.md §A.1: the pod is billed by the hour and has
# minutes, not hours, for environment setup.
#
#   bash scripts/pod_bootstrap.sh          # setup only
#   bash scripts/pod_bootstrap.sh --gate   # setup, then run gate G1
#
# Export HF_TOKEN before running if the checkpoint is gated (Gemma is).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "=== 0. GPU sanity ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: no nvidia-smi. This is not a GPU pod." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode \
           --format=csv,noheader
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM_MIB}" -lt 40000 ]; then
  echo "WARNING: ${GPU_MEM_MIB} MiB of VRAM. DESIGN.md A.2 assumes an A40 48GB" >&2
  echo "         for a 12B model in bf16 (~24GB weights + activations)." >&2
  echo "         Do NOT quantize to make it fit - A.2 calls that an unforced" >&2
  echo "         error in a paper about activation geometry." >&2
fi

echo
echo "=== 1. determinism environment ==="
# cuBLAS reads CUBLAS_WORKSPACE_CONFIG when the CUDA context is created, so it
# must be exported BEFORE python starts. ling/determinism.py also sets it at
# import time as a fallback, but exporting it here is the belt.  (DESIGN.md A.4)
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false     # deterministic + avoids fork warnings
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
grep -q CUBLAS_WORKSPACE_CONFIG ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
RC
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}  PYTHONHASHSEED=${PYTHONHASHSEED}"

echo
echo "=== 2. HuggingFace cache + auth ==="
# A.1 rule 2: "Model download is billed time. ~24GB onto a fresh pod is 5-15
# paid minutes. Use a network volume or cached template if you expect to
# restart pods." Point HF_HOME at the persistent volume if one is mounted.
for CAND in /workspace /runpod-volume /data; do
  if [ -d "$CAND" ] && [ -w "$CAND" ]; then HF_ROOT="$CAND/hf"; break; fi
done
export HF_HOME="${HF_HOME:-${HF_ROOT:-$HOME/.cache}/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1      # faster download; falls back if absent
mkdir -p "$HF_HOME"
echo "HF_HOME=${HF_HOME}  ($(df -h "$HF_HOME" | tail -1 | awk '{print $4}') free)"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is unset. Gemma checkpoints are gated - the download" >&2
  echo "         will 401. export HF_TOKEN=hf_... and accept the licence on the" >&2
  echo "         model page with the same account." >&2
fi

echo
echo "=== 3. python deps ==="
python3 -m pip install --upgrade pip --quiet
if python3 -c "import torch" 2>/dev/null; then
  echo "torch $(python3 -c 'import torch;print(torch.__version__)') already present - not reinstalling"
  grep -v '^torch==' requirements.txt > /tmp/req_notorch.txt
  python3 -m pip install --quiet -r /tmp/req_notorch.txt hf_transfer
else
  python3 -m pip install --quiet -r requirements.txt hf_transfer
fi
python3 - <<'PY'
import torch, transformers, numpy, scipy, sklearn, yaml
print("torch       ", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy       ", numpy.__version__, "| scipy", scipy.__version__,
      "| sklearn", sklearn.__version__)
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("device      ", torch.cuda.get_device_name(0))
PY

echo
echo "=== 4. ready ==="
echo "  next: python3 scripts/01_determinism_gate.py"
echo "  (DESIGN.md A.12: fork signal exactly 0, or stop.)"

if [ "${1:-}" = "--gate" ]; then
  echo
  exec python3 scripts/01_determinism_gate.py
fi
