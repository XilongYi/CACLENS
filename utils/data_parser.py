# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved


import torch
import numpy as np
from rdkit import Chem

def data_collate(reaction_feature, proteins, labels=None, device='cpu'):
    """
    Collate data into a batch

    reaction_feature : data of reaction feature 
    proteins: data of proteins
    labels: data of labels(optional: if not given,  you should use Predictor)
    device_params (torch device): Device to use.
   
    """
    # must have same length and same dim_length
    assert len(reaction_feature) == len(proteins)

    N = len(proteins)

    protein_nums = [protein.shape[0] for protein in proteins]

    proteins_new = torch.stack(proteins, dim=0).to(device)  
    reactions = torch.stack(reaction_feature, dim=0).unsqueeze(1).to(device)
    protein_nums = torch.tensor(protein_nums, dtype=torch.int).to(device)
    if labels is not None:
        labels_new = torch.tensor(labels, dtype=torch.long, device=device)

        return (reactions, proteins_new, labels_new, protein_nums)
    
    return reactions, proteins_new, protein_nums
    

def collate_fn(batch):
    """
    Args:
        batch: list of data, each atom, adj, protein, (label)
    Note:
        if label is not given, you should use Predictor
        else, you should use Trainer
    """
    collate_data = zip(*batch)
    if len(batch[0]) == 2:
        reaction_feature, proteins = collate_data
        labels = None
    elif len(batch[0]) == 3:
        reaction_feature, proteins, labels = collate_data
    else:
        raise ValueError("Wrong collate_fn input")
    return data_collate(reaction_feature, proteins, labels)



def get_shuffle_data(df, length=None, test_length=20000):
    """"
    input dataframe and return train test data
    """
    if not length:
        length = len(df)
    arr = np.arange(length)
    np.random.shuffle(arr)
    test_i, train_i = arr[:test_length], arr[test_length:]
    return df.iloc[train_i].reset_index(drop=True), df.iloc[test_i].reset_index(drop=True)