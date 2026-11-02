#!/usr/bin/env bash
# setup_server.sh -- prepare the Python environment for NEGSC on the lab server.
#
# Target: Ubuntu 22.04+ (the lab login node), a CUDA GPU, and NO sudo access.
# Everything below is installed into the home directory, so it works on a
# shared machine that was "minimized" (no python3-venv, no pip, no build tools).
#
# Why uv: it downloads a standalone Python 3.10, which means we do not need
# python3-venv/apt at all. Python 3.10 is required because the released
# notebook calls `random.sample(set, k)`, which raises on Python >= 3.11.
#
# Usage:
#   cd ~/negsc-fraud-detection
#   bash setup_server.sh
#   source ~/.venvs/negsc/bin/activate
#
# If your driver predates CUDA 12.1 (see `nvidia-smi`), read CUDA_NOTE below.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/.venvs/negsc"
# CUDA_NOTE: torch 2.3.0 wheels are published for cu118 and cu121.
#   driver >= 525.60.13  -> cu121 (default below)
#   driver <  525.60.13  -> use cu118: change the index URL to /whl/cu118
#                           and the DGL index to .../torch-2.3/cu118/repo.html
TORCH_CUDA_TAG="cu121"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
DGL_INDEX="https://data.dgl.ai/wheels/torch-2.3/${TORCH_CUDA_TAG}/repo.html"

echo "== 1. checks =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Is this a GPU node?" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader
echo
echo "GPU processes currently running (be a good citizen -- if someone is"
echo "training on a card, pick another or wait):"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
           --format=csv,noheader 2>/dev/null || echo "  (none reported)"
echo
echo "disk:  $(df -h "$HOME" | awk 'NR==2{print $4" free on "$6}')"
echo "ram:   $(free -g | awk 'NR==2{print $2" GB total, "$7" GB available"}')"
echo "cores: $(nproc)"
echo

echo "== 2. udf, git, tmux =="
for tool in git tmux curl; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "  $tool: $(command -v "$tool")"
  else
    echo "  $tool: MISSING -- install with: sudo apt install $tool"
    echo "         (tmux matters: a bare ssh session dies on disconnect and"
    echo "          takes a multi-hour training run with it)"
  fi
done

echo "== 3. uv + python 3.10 =="
export PATH="${HOME}/.local/bin:${PATH}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version
uv venv --python 3.10 "${VENV_DIR}" --allow-existing
export VIRTUAL_ENV="${VENV_DIR}"

echo "== 4. dependencies =="
uv pip install "numpy<2" "torch==2.3.0" --index-url "${TORCH_INDEX}"

# DGL: the local version tag must be pinned explicitly -- PyPI also serves
# dgl 2.2.1 (CPU build), and it wins on version comparison.
uv pip install "dgl==2.2.0+${TORCH_CUDA_TAG}" -f "${DGL_INDEX}"

# torchdata 0.8.0 pairs with torch 2.3. 0.7.1 fails on Linux with
# "cannot import name 'DILL_AVAILABLE'". The rest are DGL graphbolt's imports.
uv pip install "torchdata==0.8.0" pyyaml pydantic psutil tqdm requests

uv pip install pandas scikit-learn category_encoders networkx scipy \
  matplotlib seaborn pytest jupyterlab ipykernel

echo "== 5. CUDA library path for DGL =="
# DGL dlopens libnvrtc from site-packages/nvidia/*/lib, which the loader does
# not search. Without this: "Cannot load Graphbolt C++ library".
ACTIVATE="${VENV_DIR}/bin/activate"
if ! grep -q "DGL CUDA libraries" "$ACTIVATE"; then
  {
    echo ""
    echo "# DGL CUDA libraries: expose libnvrtc from torch's bundled nvidia packages"
    echo 'export DGLBACKEND=pytorch'
    echo '_NVIDIA_LIBS="$(find "$VIRTUAL_ENV/lib" -maxdepth 5 -type d -path "*/nvidia/*/lib" 2>/dev/null | paste -sd:)"'
    echo 'export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
    echo 'unset _NVIDIA_LIBS'
  } >> "$ACTIVATE"
  echo "appended CUDA library path to ${ACTIVATE}"
fi

echo "== 6. verify =="
export DGLBACKEND=pytorch
export LD_LIBRARY_PATH="$(find "${VENV_DIR}/lib" -maxdepth 5 -type d -path '*nvidia/*/lib' 2>/dev/null | paste -sd:)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${VENV_DIR}/bin/python" - <<'PY'
import sys
import torch, dgl, numpy, sklearn, category_encoders

print("python", sys.version.split()[0])
print("torch ", torch.__version__, "| cuda", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("dgl   ", dgl.__version__, "| numpy", numpy.__version__,
      "| sklearn", sklearn.__version__)
assert torch.cuda.is_available(), "torch cannot see the GPU"
g = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=2).to("cuda")
print("dgl graph on GPU:", g.device, "-- ok")
PY

cat <<EOF

Done. Environment: ${VENV_DIR}

Activate:   source ${VENV_DIR}/bin/activate
Then run:   cd ${REPO_DIR}
            PYTHONHASHSEED=0 python -m pytest tests/ -q
            PYTHONHASHSEED=0 python main.py --rows 200000 --chunks 1 --epochs 3 \\
                --skip-predict --run-id server_smoke

For anything longer than a few minutes, run inside tmux so an ssh disconnect
does not kill it:

  tmux new -s negsc
  # ... run the training ...
  # detach with Ctrl-b then d   |   reattach with: tmux attach -t negsc
EOF
