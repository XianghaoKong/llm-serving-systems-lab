#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/workspace/pip-cache
export TORCH_EXTENSIONS_DIR=/workspace/torch-extensions
mkdir -p /workspace/s10/environment
nvidia-smi -q > /workspace/s10/environment/nvidia-smi-q.txt
nvidia-smi topo -m > /workspace/s10/environment/topology.txt
python -m venv --system-site-packages /workspace/s10-venv
source /workspace/s10-venv/bin/activate
python -m pip install 'numpy==1.26.4' 'deepspeed==0.17.5' 'transformers==4.57.1' 'accelerate==1.11.0' 'einops==0.8.1' 'pybind11==3.0.1' 'tensorstore==0.1.76' 'nvtx==0.2.13' 'sentencepiece==0.2.1' 'tiktoken==0.11.0' 'ninja==1.13.0' 'packaging==25.0'
if [[ -d /workspace/Megatron-LM/.git ]]; then
  test "$(git -C /workspace/Megatron-LM rev-parse HEAD)" = 23e00ed0963c35382dfe8a5a94fb3cda4d21e133
else
  git clone --depth 1 --branch core_v0.14.0 https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM
fi
cd /workspace/Megatron-LM
python -m pip install --no-deps --no-build-isolation -e .
python -m pip freeze > /workspace/s10/environment/pip-freeze.txt
git rev-parse HEAD > /workspace/s10/environment/megatron-commit.txt
python -c 'import torch,deepspeed; print(torch.__version__, torch.cuda.device_count(), deepspeed.__version__)'
touch /workspace/s10/setup-complete
