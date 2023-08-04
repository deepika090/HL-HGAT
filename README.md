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
conda create -n HLHGCNN python=3.9
conda activate HLHGCNN

conda install pytorch=1.12 torchvision torchaudio pytorch-cuda=10.2 -c pytorch -c nvidia
conda install pyg=2.1 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
# https://data.pyg.org/whl/torch-1.12.1%2Bcu102.html
# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  

conda install -c conda-forge torchmetrics
conda install -c conda-forge torch-scatter
conda install pytorch-cluster -c pyg
conda install -c conda-forge timm
conda install -c anaconda networkx
conda install -c conda-forge mat73
conda install -c anaconda scipy

conda clean --all
```
