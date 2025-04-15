# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

from utils.reaction_cls import ec_get_reaction_cls_data, EC_ContrastiveReactionDataset, get_rcls_dataloader, rcls_train_epoch
from utils.utils import *
from model.cgc_framework import PLE_Reaction
from utils.builder import get_optimizer, LabelSmoothingLoss,SupConHardLoss 
from config import CFG
from model.model import ReactionClassificationTower
from utils.embedding import EmbedingModel
import pandas as pd
import torch
import os
# cfg
seed_everything(seed=42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RCls_Config:
    def __init__(self):
        self.reaction_model_name = "molformer"
        self.hidden_dim=256
        self.out_dim=256
        self.batch_size = 512
        self.p_samples=9
        self.n_samples=30
        self.temp = 0.1
        self.data = "./data/rcls/ec_reaction_data_split.csv"
        self.learning_rate = 0.8e-4 
        self.weight_decay = 1e-9

rcls_config = RCls_Config()

## model     
ple_reaction = PLE_Reaction().to(device)
embedding_model = EmbedingModel(
    protein_model='esm36', reaction_model=rcls_config.reaction_model_name)

rcls_tower = ReactionClassificationTower(rcls_config, ple_reaction=ple_reaction).to(device)
rcls_tower.load_state_dict(torch.load("./model/model_save/EF/rcls_tower.pth"))

## data
reaction_data = pd.read_csv(rcls_config.data)
rcls_train_em, rcls_train_labels, rcls_test_em, rcls_test_labels = ec_get_reaction_cls_data(
    embedding_model, reaction_data)
rcls_train_dataset = EC_ContrastiveReactionDataset(
    rcls_train_em, rcls_train_labels, p_samples=rcls_config.p_samples, n_samples=rcls_config.n_samples)
rcls_train_loader = get_rcls_dataloader(rcls_train_dataset, rcls_config.batch_size)

# loss
rcls_criterion = SupConHardLoss
#optimizer
rcls_optimizer = get_optimizer(rcls_tower, rcls_config.weight_decay, rcls_config.learning_rate)


## train
for epoch in range(1, 6):
    print('epoch: ' + str(epoch))
    print('Fine-tuning........................')	
    torch.cuda.empty_cache()
    # rcls_train
    rcls_train_loss = rcls_train_epoch(rcls_tower, rcls_train_loader, rcls_criterion, rcls_optimizer, 
                                             rcls_config.temp, rcls_config.p_samples)
    print(f"Reaction cls of EC:  Train Loss: {rcls_train_loss:.4f}")
    torch.save(rcls_tower.state_dict(), f"./model/model_save/rcls_tower_ec_{str(epoch)}_{rcls_config.reaction_model_name}.pth")
    

    
        
    
        
	 


