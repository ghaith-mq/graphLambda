# MegaDTA: Binding Affinity Prediction Using Graph Neural Networks.
Implementation of MegaDTA, a deep learning model to score the binding affinity of protein-ligand complexes in PyTorch and PyTorch Geometric.
![alt text](https://github.com/ghaith-mq/MegaDTA/blob/main/MegaDTA.png)

## Overview

We provide the implementation of the MegaDTA model in [Pytorch](https://github.com/pytorch/pytorch) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) frameworks, along with the scripts that can be used to train the model and also replicate the results. The repository is orignaized as follows:

- `models` contains:
  - `Different_GNNs` : contains various GNN models implemented in Pytorch. All possible combinations of (GCN,GAT,GIN) are provided. 
  - `MegaDTA.py` : The overall model implemented in Pytorch.


- `Data` contains:
  - `Dataset.py` : Dataset class that combines pre-computed BPS features. This dataset is to be passed to the dataloader to train the model.
  - `data.txt` : Description of the used data and benchmark. Also links to download the data are provided.
  - `refined_data2020.csv`: A **csv** file that contains **PDB** codes of protein-ligand complexes with the expiremental binding affinity.
  - `QSAR_set1.csv` , `QSAR_set1.csv`  and `coreset2016.csv` : **csv** files of used benchmarks containing  **PDB** codes of protein-ligand complexes with the expiremental binding affinity.


- `utls` contains: 
  - `remove_water.py`: This script removes water molecules from the PDB complexes.
  - `BPS_features.py` : A function that computes BPS features for the training set (**refined_set**) and test sets (**CASF16**)(**QSAR_NRC_HiQ**).
 
## Prepare the environment:

```sh
$ conda env create -f environment.yml
$ source activate myenv
$ conda env list
```

## Using the model:
- The final model can be downloaded using the [link](https://drive.google.com/file/d/1RJiA_hi6yfZP8IzH30UtnvaJvQXwjNAH/view?usp=sharing) 
- To replicate the results you need to:
  - Download the PDBbind test set "coreset" from  http://www.pdbbind.org.cn/. Also download the CSAR test sets from http://csardock.org/
  - Preprocess the PDB samples using the `remove_water.py` 
  - Generate xyz format of lignad files from existing sdf files using RDKit with the option remove H = True.
  -  Generate features.h5 file using`BPS_features.py`.
  -  To train the model, follow the steps in the training_notebook.
  - To use the model in inference mode run `use.py` passing the following arguments: 
     - directory where the test samples (PDB complexes) are located.
     - directory of computed features file of the test samples (*.h5) 
     - directory of a dataframe (csv_file) containing PDB codes of the test samples.
     - directory where the model is downloaded 
  - Run `use.py` script:
```sh
$ python use.py testset_directory testset_directory/testset.h5  testset_directory/pdb_codes.csv  model_directory.pt
```
## Results:
- **CASF16** Benchmark:
![alt text](https://github.com/ghaith-mq/MegaDTA/blob/main/results/correlation_plots_casf.png)

- **QSAR_HiQ_NRC** Benchmark:
  - **Set1**:
![alt text](https://github.com/ghaith-mq/MegaDTA/blob/main/results/cor_plots_csar1.png)
  - **Set2**:
![alt text](https://github.com/ghaith-mq/MegaDTA/blob/main/results/cor_plots_csar2.png)

## License:
 <font size = "7" >  MIT License </font>
 Copyright (c) 2022 Ghaith Mqawass


