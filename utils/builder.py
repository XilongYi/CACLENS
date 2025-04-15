# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

from .lookahead import Lookahead
import torch.nn.functional as F
from torch import nn
from .Radam import *
import torch
import os


def SupConHardLoss(model_emb, temp, n_pos):
    model_emb = model_emb.squeeze(2)
    # L2 normalize every embedding
    features = F.normalize(model_emb, dim=-1, p=2)
    # Transpose features to [batch_size, output_dim, n_all] for batch dot product
    features_T = torch.transpose(features, 1, 2)
    # Select the anchor embeddings (first element in the sequence)
    anchor = features[:, 0]
    # Compute the dot product between the anchor and all features, scaled by temperature
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T) / temp  
    anchor_dot_features = anchor_dot_features.squeeze(1)  # Reshape to [batch_size, n_all]
    # Subtract max logits (which is 1/temp) for numerical stability
    logits = anchor_dot_features - 1 / temp  
    # Exponentiate the logits, excluding the dot product of the anchor with itself
    # exp_logits has shape [batch_size, n_pos + n_neg]
    exp_logits = torch.exp(logits[:, 1:])
    # Sum of exponentiated logits, scaled by the number of positive samples
    exp_logits_sum = n_pos * exp_logits.sum(1)  # Shape: [batch_size]
    # Sum over the logit values corresponding to positive pairs
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1)  
    # Compute the log probability
    log_prob = (pos_logits_sum - exp_logits_sum) / n_pos  
    # Compute the final loss (negative mean log probability)
    loss = -log_prob.mean()
    return loss

class LabelSmoothingLoss(nn.Module):
    r'''CrossEntropyLoss with Label Smoothing.
    Args:
        classes (int): number of classes.
        smoothing (float): label smoothing value.
        dim (int): the dimension to sum.
    Shape:
        - pre: :math:`(N, C)` where `C = 2`, predicted scores.
        - target: :math:`(N)`, ground truth label
    Returns:
        loss
    '''
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def xavier_initialization(model, state_dict_path=None, trainfirst=True):
    if state_dict_path is None and trainfirst:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def load_state_dict(model, model_savepath, layer_keys=None, requires_grad=True):
    model.load_state_dict(torch.load(model_savepath), strict=True)
    if layer_keys is not None:
        pretrained_weights = torch.load(model_savepath)
        layer_state_dict = {k: v for k, v in pretrained_weights.items() if k in layer_keys}
        model.load_state_dict(layer_state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = requires_grad

def get_optimizer(model, weight_decay, lr):
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
    optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    return optimizer



    

    


    
