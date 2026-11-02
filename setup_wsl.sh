#!/usr/bin/env bash
# setup_wsl.sh -- prepare the Python environment for running NEGSC under WSL2.
#
# ============================================================================
# WHY WSL AT ALL
# ============================================================================
# DGL publishes CUDA wheels for Linux only. The index
# https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html contains 25 wheels, all
# manylinux -- zero for Windows. So on native Windows you can have
# torch 2.3.0+cu121 reporting cuda True and still get
#   DGLError: Device API cuda is not enabled
# on the first graph.to('cuda'). Under WSL2 the Linux wheel exists and the GPU
# is passed through, which is why this script exists.
#
# ============================================================================
# WHAT IT DOES
# ============================================================================
# 1. Checks WSL2, GPU passthrough and the repository location.
# 2. Aligns git's line-ending settings with the Windows checkout
#    (core.autocrlf=true), otherwise every file looks modified inside WSL.
# 3. Installs `uv` (no sudo) and uses it to fetch Python 3.10.
# 4. Creates the venv OUTSIDE /mnt -- /mnt/d is a Windows mount and is
#    dramatically slower for importing thousands of small module files.
# 5. Installs torch 2.3.0+cu121, DGL 2.2.0+cu121, and the rest.
# 6. Verifies that a DGL graph can actually live on the GPU.
#
# Usage:
#   cd /mnt/d/AnomalyDetection_Lab/negsc-fraud-detection
#   bash setup_wsl.sh
#   source ~/.venvs/negsc/bin/activate
#
# The environment lives at ~/.venvs/negsc (Linux filesystem). The code stays
# on /mnt/d so Windows and WSL share one working tree and one git history.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/.venvs/negsc"
PY_VERSION="3.10"

echo "== 1. checks =="
if ! grep -qi microsoft /proc/version; then
  echo "ERROR: this does not look like WSL." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found -- GPU is not passed through to WSL." >&2
  echo "       Update the Windows NVIDIA driver (it carries the WSL library)." >&2
  exit 1
fi
nvidia-smi -L | head -1

case "$REPO_DIR" in
  /mnt/*) echo "note: running from $REPO_DIR (Windows mount). Code stays here;"
          echo "      the venv goes to ${VENV_DIR} for import speed." ;;
esac

echo "== 2. git line endings =="
cd "$REPO_DIR"
# Windows checked the tree out with CRLF; without this, `git status` inside
# WSL reports every file as modified and `git diff` is unusable.
git config core.autocrlf true
git config core.filemode false
echo "core.autocrlf=$(git config --get core.autocrlf)"

echo "== 3. uv + python ${PY_VERSION} =="
export PATH="${HOME}/.local/bin:${PATH}"
# The default uv HTTP timeout is short; this connection stalls intermittently.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

echo "== 4. venv at ${VENV_DIR} =="
uv venv --python "${PY_VERSION}" "${VENV_DIR}" --allow-existing

# uv pip needs the target venv; VIRTUAL_ENV is the simplest way to point it.
export VIRTUAL_ENV="${VENV_DIR}"

echo "== 5. dependencies =="
uv pip install "numpy<2" "torch==2.3.0" \
  --index-url https://download.pytorch.org/whl/cu121

# DGL 2.2.0+cu121 is the Linux build paired with torch 2.3. The local version
# tag must be pinned explicitly: PyPI also serves dgl 2.2.1 (CPU), and pip
# prefers 2.2.1 over 2.2.0+cu121 by version comparison.
uv pip install "dgl==2.2.0+cu121" \
  -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# DGL's graphbolt imports these at load time.
#
# torchdata version: torch 2.3 pairs with torchdata 0.8.0. On Linux, 0.7.1
# fails with "cannot import name 'DILL_AVAILABLE' from
# torch.utils.data.datapipes.utils.common" because torch 2.3 removed that
# symbol. (On the Windows CPU env 0.7.1 imports fine -- the two platforms
# differ, so do not copy the pin across.)
uv pip install "torchdata==0.8.0" pyyaml pydantic psutil tqdm requests

uv pip install pandas scikit-learn category_encoders networkx scipy \
  matplotlib seaborn pytest jupyterlab ipykernel

echo "== 6. CUDA library path for DGL =="
# DGL 2.2.0+cu121 dlopens libnvrtc.so.12, which lives in the nvidia packages
# inside site-packages -- somewhere the system loader does not search. Without
# this, `import dgl` fails with:
#   OSError: libnvrtc.so.12: cannot open shared object file
#   ImportError: Cannot load Graphbolt C++ library
# Appended to the venv's activate script so every activation sets it, guarded
# so re-running this script does not append it twice.
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

echo "== 7. verify =="
# The venv is not activated in this shell, so set the path here too.
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

Activate and run:
  source ${VENV_DIR}/bin/activate
  cd ${REPO_DIR}
  PYTHONHASHSEED=0 python -m pytest tests/ -q

  # GPU smoke (uses the 181 MB cache, not the 6 GB raw file)
  PYTHONHASHSEED=0 python main.py --rows 200000 --chunks 1 --epochs 3 \\
      --skip-predict --run-id wsl_smoke

Expected: device=cuda:0 in the run line, and a manifest in outputs/ with
"dgl_cuda": true.

If WSL runs out of memory on the full cache, give it more than the default
~50% of host RAM by creating %UserProfile%\\.wslconfig on Windows:
  [wsl2]
  memory=10GB
then run: wsl --shutdown
EOF