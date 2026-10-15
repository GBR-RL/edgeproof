#!/usr/bin/env bash
# A second environment for the GPU work: TensorRT and NVIDIA ModelOpt. ModelOpt's ONNX extra pins an
# older onnx and brings onnxruntime-gpu, which would replace the packages the CPU results were
# measured with, so it gets its own venv. Same torch and Ultralytics as the CPU environment.
# Compute nodes have no internet: everything Ultralytics would install at run time is listed here.
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
[ -d "$BASE/venv-gpu" ] || python -m venv "$BASE/venv-gpu"
source "$BASE/venv-gpu/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.175 onnxslim==0.1.98 \
  "nvidia-modelopt[onnx]==0.47.0" huggingface_hub tensorrt==11.4.0.106
pip install -q -e "$BASE/repo[train]"
python -c "import torch, tensorrt, modelopt, ultralytics; print(torch.__version__, torch.version.cuda, tensorrt.__version__, modelopt.__version__, ultralytics.__version__)"
