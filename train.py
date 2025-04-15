# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

from utils.ec_number_pre import get_ec_dataloader, ec_train_epoch
from utils.reaction_cls import get_reaction_cls_data, ContrastiveReactionDataset, get_rcls_dataloader, rcls_train_epoch
from utils.utils import *
from utils.reaction_feasibility_pre import rpre_train_epoch, rpre_test, rpre_predict, get_reaction_pre_data, TestReader, TrainReader
from model.cgc_framework import PLE_Protein,PLE_Reaction
from utils.builder import get_optimizer, LabelSmoothingLoss,SupConHardLoss 
from config import CFG
from model.model import ReactionClassificationTower,ECPredictionTower,ReactionPredictionTower
from utils.embedding import EmbedingModel
from utils.tools import get_ec_id_dict, compute_distance
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import torch
import os
# cfg
seed_everything(seed=42)

CFG = CFG()
ec_config = CFG.ec
rcls_config = CFG.r_cls
rpre_config = CFG.r_pre

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## model     
ple_protein = PLE_Protein().to(device)
ple_reaction = PLE_Reaction().to(device)

embedding_model = EmbedingModel(
    protein_model=CFG.protein_em_model, reaction_model=CFG.reaction_em_model)

ec_tower = ECPredictionTower(
	ec_config.hid_dim, ec_config.out_dim, ple_protein=ple_protein).to(device)

rcls_tower = ReactionClassificationTower(rcls_config, ple_reaction=ple_reaction).to(device)

rpre_tower = ReactionPredictionTower(
    rpre_config, ple_protein=ple_protein, ple_reaction=ple_reaction).to(device)

models = [ec_tower, rcls_tower, rpre_tower]
for tower in models:
    xavier_initialization(tower, trainfirst=True)

## data
# ec
id_ec, ec_id_dict = get_ec_id_dict(ec_config.train_data)
ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}

# loading protein embedding for dist map
ec_train_data = pd.read_csv(ec_config.train_data, sep='\t')
ec_test_data = pd.read_csv(ec_config.test_data, sep='\t')
compute_distance(embedding_model, 'split100')
protein_emb = pickle.load(open('./data/distance_map/split100_em.pkl', 'rb')).to(device=device)
dist_map = pickle.load(open('./data/distance_map/split100.pkl', 'rb')) 

ec_train_loader = get_ec_dataloader(dist_map, id_ec, ec_id, ec_config)
print("The number of unique EC numbers: ", len(dist_map.keys()))

# rcls
reaction_data = pd.read_csv(rcls_config.data)
rcls_train_em, rcls_train_labels, rcls_test_em, rcls_test_labels = get_reaction_cls_data(
    embedding_model, reaction_data)
rcls_train_dataset = ContrastiveReactionDataset(
    rcls_train_em, rcls_train_labels, p_samples=rcls_config.p_samples, n_samples=rcls_config.n_samples)
rcls_train_loader = get_rcls_dataloader(rcls_train_dataset, rcls_config.batch_size)

#rpre
rpre_p_test, rpre_n_test, rpre_p_train, rpre_n_train = get_reaction_pre_data(rpre_config)


rpre_traindata = TrainReader(embedding_model, positive_data=rpre_p_train, unlabel_data=rpre_n_train)
rpre_testdata = TestReader(embedding_model, positive_data=rpre_p_test, unlabel_data=rpre_n_test)

rpre_val_dataloader = DataLoader(rpre_testdata, batch_size=rpre_config.batch_size, collate_fn=collate_fn)

# loss
ec_criterion = SupConHardLoss
rcls_criterion = SupConHardLoss
rpre_criterion = LabelSmoothingLoss(classes=2, smoothing=0.05)

#optimizer
ec_optimizer = get_optimizer(ec_tower, CFG.weight_decay, ec_config.learning_rate)
rcls_optimizer = get_optimizer(rcls_tower, CFG.weight_decay, rcls_config.learning_rate)
rpre_optimizer = get_optimizer(rpre_tower, CFG.weight_decay, rpre_config.learning_rate)


## train
for epoch in range(1, CFG.epochs_1+CFG.epochs_2+1):
    print('epoch: ' + str(epoch))
    print('Training........................')	
    torch.cuda.empty_cache()
    # ec_train
    ec_train_loss = ec_train_epoch(ec_tower, ec_train_loader, ec_criterion, ec_optimizer, ec_config, 
                                   protein_emb, ec_id_dict, id_ec, ec_id, epoch, CFG.epochs_1+CFG.epochs_2)
    print(f"EC Loss: {ec_train_loss}")

    # rcls_train
    rcls_train_loss = rcls_train_epoch(rcls_tower, rcls_train_loader, rcls_criterion, rcls_optimizer, 
                                             rcls_config.temp, rcls_config.p_samples)
    print(f"Reaction cls:  Train Loss: {rcls_train_loss:.4f}")
    
    if epoch == CFG.epochs_1:
        for optimizer in [ec_optimizer,rcls_optimizer]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10 

    if epoch >= CFG.epochs_1:
        # rpre_train
        rpre_train_dataloader = DataLoader(rpre_traindata, batch_size=rpre_config.batch_size, 
                                            shuffle=True, collate_fn=collate_fn)
        
        rpre_loss_total = rpre_train_epoch(rpre_tower, rpre_train_dataloader, rpre_criterion, rpre_optimizer)
        rpre_auc_dev, rpre_precision, rpre_recall, rpre_prc_dev = rpre_test(rpre_tower, rpre_val_dataloader)
        print(f"Reaction pre:  Loss: {rpre_loss_total:.4f}, AUC: {rpre_auc_dev:.4f}, Precision: {rpre_precision:.4f}, Recall: {rpre_recall:.4f}, PRC AUC: {rpre_prc_dev:.4f}")
    if epoch == CFG.epochs_1+CFG.epochs_2:
        torch.save(ec_tower.state_dict(), f"./model/model_save/ec_tower_{CFG.protein_em_model}+{CFG.reaction_em_model} .pth")
        torch.save(rcls_tower.state_dict(), f"./model/model_save/rcls_tower_{CFG.protein_em_model}+{CFG.reaction_em_model} .pth")
        torch.save(rpre_tower.state_dict(), f"./model/model_save/rpre_tower_{CFG.protein_em_model}+{CFG.reaction_em_model} .pth")


    
        
    
        
	 


