# ENVPOL-D-25-07213 论文模型

## Environment Setup:
```shell

conda create -n env python=3.12
conda activate env
pip install lightning -i https://mirrors.aliyun.com/pypi.simple
```
- Our model backbone network is implemented based on PyTorch Lightning and is fully compatible with standard PyTorch network architectures.
- Data input format: [B, T, N, C]


