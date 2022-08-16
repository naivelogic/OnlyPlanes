#!/bin/bash

echo "=================================================="
echo "STARTING: OnlyPlanes (OP) Python Environment Installation"

conda create --name OP python=3.8.12
eval "$(conda shell.bash hook)"
conda activate OP
pip install -r requirements.txt
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html


## if using Jupyter Notebooks create custom jupyter kernel for OnlyPlanes detectron2
python -m ipykernel install --user --name=OP
python -m pip install setuptools==59.5.0

echo "=================================================="
echo "OnlyPlanes Python Environment Installation Complete"
echo "Run: conda activate OP"
echo "=================================================="