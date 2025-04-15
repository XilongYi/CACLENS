# Title

This repository contains the cross-attention and contrastive learning-based enzyme screening (**CACLENS**) system as described in the paper CACLENS: a Multitask Deep Learning System for Enzyme Discovery
.

# Requirements
-   OS support: Linux
-   Python version: >=3.8

A Linux workstation with a GPU is essential for deploying CACLENS. The final **CACLENS** model was trained on a NVIDIA A100-PCIE-40GB GPU, taking approximately 1.5 days.
# Dependencies

The code has been tested in the following environment:



| **Package**        | **Version** |
|--------------------|------------|
| PyTorch           | 2.3.1      |
| CUDA              | 11.6.1     |
| RDKit             | 2024.3.5   |
| Scikit-learn      | 1.3.2      |
| Triton            | 2.3.1      |
| Transformers      | 4.44.2     |
| Huggingface Hub   | 0.24.6     |
| ONNX Runtime      | 1.19.2     |
| Pandas            | 2.0.3      |
| Tmap              | 1.0.6      |

The following commands will clone the **CACLENS** project and download the necessary data for further experiments or inference.

```bash
git clone https://.git
cd 
wget https://drive.google.com/file/d/1e7IOSnQfHxII4KpOUitxlyrbBy-09DPm/view?usp=sharing
tar -zxvf checkpoints.tar.gz
```
# Training

Execute `train.py` and customize the configuration in `config.py`.
```bash
   python train.py
```
# Predicting


You can use `predict.py` for **screening functional enzymes based on arbitrary chemical reactions**.

`predict.py`  allows users to:

- Filter enzymes using EC numbers based on the reactant and product.

- Filter enzymes using a provided EC number.

- Perform direct enzyme screening.

### Usage

Run the script with the following parameters:

 
```bash

python  predict.py  --reactant  <reactant>  --product  <product>  --screening_type  <screening_type>  --seq  <enzyme_file_or_sequence> --model <enzyme screening model>  [--ec_num <EC_number>]

```
The functions of different parameters are as follows:


  


| **Argument**     | **Required** | **Description**                                      | **Example**     |
|------------------|-------------|------------------------------------------------------|-----------------|
| `--reactant`    | Yes         | Input reactant (chemical formula, SMILES, etc.)    | `"COc1cc(C=CC(=O)[O-])cc(OC)c1O"`     |
| `--product`     | Yes         | Input product (chemical formula, SMILES, etc.)     | `"COc1cc(C=CC(=O)OCC[N+](C)(C)C)cc(OC)c1O"`      |
| `--screening_type`        | Yes         | Screening type (`re2ec`, `ec`, or `direct`)           | `"re2ec"`         |
| `--seq`  | Yes         | Path to a CSV file containing enzyme sequences or a direct sequence string | `"enzymes.csv"` |
| `--model`  | Yes         | Four models trained with different encoding methods: `'EU'`, `'EF'`, `'TU'`, `'TF'`. | `'TF'` |
| `--ec_num`   | No (Required for `ec` screening type) | EC number for filtering | `"3.1.1.*"`  |

 

### Screening Types
CACLENS supports three different screening types using `--screening_type`:

| **screening type**   | **Description**                                                                 | **Required Parameters**                        |
|-----------|---------------------------------------------------------------------------------|----------------------------------------------|
| `re2ec`     | Filter enzymes using EC numbers based on the reactant and product. | `--reactant`, `--product`, `--seq` |
| `ec`      | Filters candidate enzymes based on the given EC number.                          | `--reactant`, `--product`, `--seq`, `--ec_num`               |
| `direct`  | Directly screens all enzymes in the given database.                             | `--reactant`, `--product`, `--seq`   |

### Candidate Enzyme Input
You can input a single enzyme sequence directly using `--seq`.  
If `--seq` is set to a CSV file path, the file must follow the specified format and be saved in the `candidate_enzymes` folder.

For example, if you input `--seq example.csv`, the file `example.csv` must follow the format below:

***example.csv***
| **Protein** | **ID** (Optional) |
|------------|--------------------------|
| SEQ1       | ID1                      |
| SEQ2       | ID2                      |
| SEQ3       | ID3                      |
| ...        | ...                      |
### Reaction Feasibility Prediction Model 
The `--model` parameter specifies the model used for the final reaction feasibility prediction (Reaction Prediction Tower). We applied four different encoding combinations based on protein sequences and small molecule SMILES and trained four corresponding models. The AUC values for these models are as follows:
| Model | Encoding Combination       | AUC  |
|--------|----------------------------|------|
| EU     | ESM2 + UniMol              |      |
| EF     | ESM2 + MolFormer           |      |
| TU     | ProtT5 + UniMol                 |      |
| TF     | ProtT5 + MolFormer              |      |

After a few minutes of calculation, you will find the result in the **result** folder.


# Web Server
  For researchers who lack the hardware to deploy **CACLENS**, we provide an easy-to-use web server:
  
# License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  
 **License Details:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  


# Contact
If you have any questions or suggestions, please contact **[yixilong2023@sinh.ac.cn](mailto:yixilong2023@sinh.ac.cn)**.  

You can also send us your data, and we will process the computation and return the results to you.
  

Please see the file LICENSE for details about the "MIT" license which covers this software and its associated data and documents.
