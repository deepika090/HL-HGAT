# HL-HGAT
Heterogeneous Graph Convolutional Neural Network via Hodge-Laplacian

<picture>
<!--  <source media="(prefers-color-scheme: dark)" srcset="YOUR-DARKMODE-IMAGE"> -->
<!--  <source media="(prefers-color-scheme: light)" srcset="YOUR-LIGHTMODE-IMAGE"> -->
 <img alt="YOUR-ALT-TEXT" src="Architecture.png">
</picture>

This project introduces a novel approach to transform a traditional graph into a simplex graph, where nodes, edges, and higher-order interactions are characterized by different-dimensional simplices. We propose the Hodge-Laplacian Heterogeneous Graph Attention Network (HL-HGAT), which enables simultaneous learning of features on different-dimensional simplices.

In this HL-HGAT package, we provide the transformation of the original graph to a simplex graph. Then we provide a detailed implementation of the proposed model. HL-HGAT is built using PyG and Pytorch.

## Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```
