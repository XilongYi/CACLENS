# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

from .contrastive import mine_hard_negative,MultiPosNeg_dataset_with_mine_EC,get_dist_map
from torch.cuda.amp import GradScaler, autocast
import torch
import os
from .contrastive.distance_map import *
from .contrastive.evaluate import *


scaler = GradScaler()

def ec_train(model, config, train_loader, optimizer, device, ec_criterion):
    model.train()
    total_loss = 0.

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast():
            model_emb = model(data.to(device=device))
            model_emb = model_emb.squeeze(2)
            loss = ec_criterion(model_emb, config.temp, config.n_pos)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        message = "EC Prediction Tower: Training Progress {}/{}, loss: {:.5f}"
        info_message(message, batch, len(train_loader), total_loss/(batch + 1), end="\r")

    return total_loss/(batch + 1)

def get_ec_dataloader(dist_map, id_ec, ec_id, config):
    params = {
        'batch_size': config.batch_size,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 100)
    train_data = MultiPosNeg_dataset_with_mine_EC(
        id_ec, ec_id, negative, config.n_pos, config.n_neg, em_model=config.protein_em_model)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader

def ec_train_epoch(model, train_loader, ec_criterion, optimizer, config, protein_emb, ec_id_dict, id_ec, ec_id,
                    epoch, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epoch % config.adaptive_rate == 0 and epoch != epochs + 1:
    # sample new distance map
        dist_map = get_dist_map(
            ec_id_dict, protein_emb, device, model=model)
        train_loader = get_ec_dataloader(dist_map, id_ec, ec_id, config)

    train_loss = ec_train(model, config, train_loader, optimizer, device, ec_criterion)
    
    return train_loss

    
