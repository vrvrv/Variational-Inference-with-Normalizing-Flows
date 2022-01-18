<h1 align="center">
  <b>Variational Inference with Normalizing Flows</b><br>
</h1>

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-blue.svg" /></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
    <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
    <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</p>

## Installation
```bash
https://github.com/vrvrv/Variational-Inference-with-Normalizing-Flows.git
cd Variational-Inference-with-Normalizing-Flows

# pip install -r requirements.txt
```

## Train
You can find configuration files at [configs/experiment/](configs/experiment).
In our code, [wandb](https://wandb.ai/) is the default logger. So, before running code, please sign up wandb.

### Training
If you want to control the number of hidden dimension, add `model.D=<hidden_dim>`.
```bash
python train.py experiment=mnist_nfvae model.D=10
```

### References
- [Variational Inference with Normalizing Flows, ICML](https://arxiv.org/abs/1505.05770)