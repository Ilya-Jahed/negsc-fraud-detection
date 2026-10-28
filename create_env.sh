#!/usr/bin/env bash
# Create the Python 3.10 reference environment (negsc-ref).
#
# Why 3.10: the released notebook calls `random.sample(set, k)`, which raises on
# Python >= 3.11. With 3.10 the notebook runs as published (apart from the one
# syntax repair in reference/REPAIR_DIFF.md), which is what makes the parity
# gate a fair comparison rather than two patched implementations compared.
#
# Why the pinned extras: DGL 2.2.1's `graphbolt` package imports torchdata,
# yaml and pydantic at load time, and torchdata >= 0.10 removed `datapipes`.
# Without these pins `import dgl` fails with "No module named torchdata.datapipes".
#
# Usage:  bash create_env.sh
# Then:   conda activate negsc-ref

set -euo pipefail
ENV=negsc-ref

echo "== creating ${ENV} (python 3.10) =="
conda create -n "$ENV" python=3.10 -y

echo "== torch 2.3.0 + CUDA 12.1 =="
conda run -n "$ENV" python -m pip install "numpy<2" "torch==2.3.0" \
  --index-url https://download.pytorch.org/whl/cu121

echo "== dgl 2.2.1 (cu121 wheel for torch 2.3) =="
conda run -n "$ENV" python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

echo "== dgl runtime dependencies (see header) =="
conda run -n "$ENV" python -m pip install "torchdata==0.7.1" pyyaml pydantic psutil tqdm requests

echo "== project dependencies =="
conda run -n "$ENV" python -m pip install pandas scikit-learn category_encoders \
  networkx scipy matplotlib seaborn pytest

echo "== verify =="
conda run -n "$ENV" python -c "
import sys, torch, dgl, numpy, sklearn, category_encoders
print('python', sys.version.split()[0])
print('torch ', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('dgl   ', dgl.__version__, '| numpy', numpy.__version__,
      '| sklearn', sklearn.__version__)
assert torch.ones(1).cuda().device.type == 'cuda', 'GPU tensor allocation failed'
print('GPU tensor allocation: ok')
"