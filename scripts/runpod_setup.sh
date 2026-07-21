#!/usr/bin/env bash
# Idempotent RunPod setup. Safe to re-run on every pod restart.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/apr-heart-sounds}"
cd "$REPO_DIR"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU visible"

# --- deps -------------------------------------------------------------------
# CRITICAL: the RunPod PyTorch image already ships torch built against its CUDA
# runtime. Installing torch from PyPI here would pull the CPU wheel and silently
# break GPU training. So we strip torch out of requirements.txt.
if [ ! -f .deps-installed ]; then
  echo "=== installing dependencies (skipping torch) ==="
  grep -viE '^\s*(torch|#|$)' requirements.txt > /tmp/req-notorch.txt
  pip install --no-cache-dir --upgrade pip
  pip install --no-cache-dir -r /tmp/req-notorch.txt
  pip install --no-cache-dir -e .
  touch .deps-installed
else
  echo "=== dependencies already installed (delete .deps-installed to force) ==="
fi

# --- verify -----------------------------------------------------------------
echo "=== verifying torch sees the GPU ==="
python -c "
import torch
print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device: {torch.cuda.get_device_name(0)}')
    print(f'  cuda  : {torch.version.cuda}')
else:
    raise SystemExit('CUDA NOT AVAILABLE -- training will fall back to CPU')
"

echo "=== verifying optional audio deps ==="
python -c "import librosa, pywt, shap, soundfile; print('librosa/pywt/shap/soundfile OK')"

echo "=== running unit tests ==="
python -m pytest tests/ -q --tb=short

echo ""
echo "=========================================================="
echo " Ready.  cd $REPO_DIR"
echo "   bash scripts/run_all.sh --skip-ablations"
echo "=========================================================="