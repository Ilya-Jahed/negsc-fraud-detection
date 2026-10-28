set -e
echo "== creating negsc-ref (python 3.10) =="
conda create -n negsc-ref python=3.10 -y
echo "== torch 2.3.0+cu121 =="
conda run -n negsc-ref pip install "numpy<2" "torch==2.3.0" --index-url https://download.pytorch.org/whl/cu121
echo "== dgl (cu121, torch 2.3) =="
conda run -n negsc-ref pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html || \
  conda run -n negsc-ref pip install dgl -f https://data.dgl.ai/wheels/repo.html
echo "== rest =="
conda run -n negsc-ref pip install pandas scikit-learn category_encoders networkx scipy matplotlib seaborn pytest
echo "== verify =="
conda run -n negsc-ref python -c "import sys,torch,dgl,numpy;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);print('dgl',dgl.__version__,'numpy',numpy.__version__)"
