# MegaDTA: Binding Affinity Prediction Using Graph Neural Networks.
Implementation of MegaDTA, a deep learning model to score the binding affinity of protein-ligand complexes in PyTorch and PyTorch Geometric.
![alt text](https://github.com/ghaith-mq/MegaDTA/blob/main/MegaDTA.png)

## Overview

We provide the implementation of the MegaDTA model in [Pytorch](https://github.com/pytorch/pytorch) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) frameworks, along with the scripts that can be used to train the model and also replicate the results. The repository is orignaized as follows:

- `models` contains:
  -  various GNN models implemented in Pytorch. All possible combinations of (GCN,GAT,GIN) are provided. 
  - `MegaDTA.py` : The overall model implemented in Pytorch.


- `Data` contains:
  - `Dataset.py` : Dataset class that combines pre-computed BPS features. This dataset is to be passed to the dataloader to train the model.
  - `data.txt` : Description of the used data and benchmark. Also links to download the data are provided.
  - `refined_data2020.csv`: A **csv** file that contains **PDB** codes of protein-ligand complexes with the expiremental binding affinity.
  - `QSAR_set1.csv` , `QSAR_set1.csv`  and `coreset2016.csv` : **csv** files of used benchmarks containing  **PDB** codes of protein-ligand complexes with the expiremental binding affinity.


- `BPS_features.py` : A python script that computes BPS features. 
 
## Prepare the environment:

```sh
$ conda env create -f environment.yml
$ source activate myenv
$ conda env list
```

## Using the model:
- The final models can be downloaded using the [link](https://drive.google.com/file/d/1RJiA_hi6yfZP8IzH30UtnvaJvQXwjNAH/view?usp=sharing) 
- To replicate the results you need to:
  - Download the testset "coreset" from  [link](https://drive.google.com/file/d/1RQ3dR0CmDiIIQDkOZ_0LdlF8yfgALF6s/view?usp=sharing) and place it in the same directory of the notebook. The downloaded folder contains the coreset from PDBbind with BPS features precomputed using `BPS_features.py` and stored `*.h5 file` . You need to load paths to the work directory and *.h5 file and coreset2016.csv file in the notebook MegaDTA-Use. Some preprcessing was already carried out to the original data :  
  - Preprocessed the PDB samples by removing water molecules. 
  - Generated xyz format of lignad files from existing sdf files using RDKit with the option remove H = True.
  - Generated features.h5 file using`BPS_features.py`.
  To use the model run `MegaDTA-Use.ipynb` and inside the notebook you should pass the following arguments: 
     - directory where the test samples (PDB complexes) are located.
     - directory of computed features file of the test samples (*.h5) 
     - directory of a dataframe (csv_file) containing PDB codes of the test samples.
     - directory where the model is downloaded 

## License:
 <font size = "7" >  MIT License </font>
 Copyright (c) 2022 Ghaith Mqawass


