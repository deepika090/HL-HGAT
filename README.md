# HL-HGAT
Heterogeneous Graph Convolutional Neural Network via Hodge-Laplacian

<picture>
 <img alt="Model Architecture" src="NM_Architecture.pdf">
</picture>

This project introduces a novel approach to transform a traditional graph into a simplex graph, where nodes, edges, and higher-order interactions are characterized by different-dimensional simplices. We propose the Hodge-Laplacian Heterogeneous Graph Attention Network (HL-HGAT), which enables simultaneous learning of features on different-dimensional simplices.

In this HL-HGAT package, we provide the transformation of the original graph to a simplex graph. Then we provide a detailed implementation of the proposed model. HL-HGAT is built using PyG and Pytorch.

## Python environment setup with Conda
cu102 should be replaced by the specific CUDA versions!
```bash
conda create -n HLHGCNN python=3.9
conda activate HLHGCNN
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install -y torch-cluster==1.6.0     -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-scatter==2.0.9     -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-sparse==0.6.15      -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu102.html

conda install -c conda-forge timm
conda install -c anaconda networkx
conda install -c conda-forge mat73
conda install -c conda-forge torchmetrics
conda clean --all
```


## Results
An example of the simulation result for the traveling salesman problem. We visualize the averaged node and edge features after every two convolutional layers in the HL-HGAT. The ground truth is displayed at the center of the figure. The grey and red color bars represent node and edge features, respectively. The code of figure generation is included in TSP_VISUALIZATION.ipynb.

<picture>
 <img alt="results" src="tsp_trend.pdf">
</picture>


## Usage
```bash
# train peptide func (pyr: HL-HGAT without pooling; attpool: HL-HGAT)
python main_pepfunc_HL_HGCNN_dense_int3_pyr.py
python main_pepfunc_HL_HGCNN_dense_int3_attpool.py

# train zinc
python main_zinc_HL_HGCNN_dense_int3_pyr.py
python main_zinc_HL_HGCNN_dense_int3_attpool.py

# train cifar10 superpixel
python main_cifar10SP_HL_HGCNN_dense_int3_pyr.py
python main_cifar10SP_HL_HGCNN_dense_int3_attpool.py

# train TSP
python main_TSP_HL_HGCNN_dense_int3_pyr.py
```





