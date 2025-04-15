# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

from utils.utils import *
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
from .Radam import *
import numpy as np
import pandas as pd
import torch
import os


def rpre_train_epoch(rpre_model, rpre_train_dataloader, rpre_criterion,
                      optimizer, device_params=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    rpre_model.train()
    loss_total = 0
    scaler = GradScaler()
    for i, data_pack in enumerate(rpre_train_dataloader):
        optimizer.zero_grad()
        reactions, proteins, labels, _ = d2D(data_pack, device_params)
        with autocast():
            output = rpre_model(reactions, proteins)
            loss = rpre_criterion(output, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total += loss.detach().cpu().numpy()
        message = 'Reaction Prediction Tower: Training Progress {}/{}, loss: {:.5f}, loss_total: {:.5f}'
        info_message(message, i, len(rpre_train_dataloader), loss, loss_total, end="\r")

    return loss_total


def rpre_test(rpre_model, pre_val_dataloader, device_params=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        rpre_model.eval()
        T, Y, S ,P = [], [], [], []
        with torch.no_grad():
            for _, data_pack in enumerate(pre_val_dataloader):
                reactions, proteins_new, labels_new, _ = d2D(data_pack, device_params)
                predicted_interaction = rpre_model(reactions, proteins_new)
                correct_labels = labels_new.to('cpu').data.numpy()
                ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
                predicted_labels = np.argmax(ys, axis=1)
                predicted_scores = ys[:, 1]
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                P.extend(proteins_new)
        AUC = roc_auc_score(T, S)
        Precision = precision_score(T, Y)
        Recall = recall_score(T, Y)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, Precision, Recall, PRC

def rpre_predict(rpre_model, predict_dataloader, device_params=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        rpre_model.eval()
        Y, S = [], []
        with torch.no_grad():

            for i, data_pack in enumerate(predict_dataloader):
                reactions, proteins_new = d2D(data_pack, device_params)
                reactions = reactions.to(torch.float32)
                proteins_new = proteins_new.to(torch.float32)
                predicted_interaction = rpre_model(reactions, proteins_new)

                ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
                predicted_labels = np.argmax(ys, axis=1)
                predicted_scores = ys[:, 1]
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                message = 'Reaction Prediction Tower: Progressing Prediction {}/{}'
                info_message(message, i, len(predict_dataloader), end="\r")
        return Y, S

class PredictReader(torch.utils.data.Dataset):
    r'''Read the dataset for prediction
    Args:
        data (pd.DataFrame): Predict dataset.
    '''

    def __init__(self, embeding_model, data) -> None:
        super().__init__()
        self.data = data
        self.embeding_model = embeding_model
  
    def checkpath(self, path):
        """
        Check if the path exists
        """
        if os.path.exists(path): return path
        raise Exception(f"Wrong path: {path}, this path does not exist")

    def data_embedding(self, data):
        assert isinstance(data, pd.core.series.Series), "error type"
        
        smiles = data["Reaction"]
        sequence = data["Protein"]
        unimol_repr = self.embeding_model.smiles_embedding(smiles)
        protein = self.embeding_model.sequence_embedding(sequence=sequence)
        return (unimol_repr, protein)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        res = self.data_embedding(self.data.iloc[idx])

        return res

class TestReader(PredictReader):
    r'''Read the dataset for testing
    Args:
        data (pd.DataFrame): Test dataset.
        positive_data (pd.DataFrame): positive dataset.
        unlabeled_data (pd.DataFrame): unlabeled dataset.
    '''
    def __init__(self, embeding_model,data=None,positive_data=None, unlabel_data=None):
        if data is None:
            assert positive_data is not None and unlabel_data is not None,'wrong input'
        if data is not None: 
            assert positive_data is None and unlabel_data is None, 'wrong input'
        if data is not None:
            self.positive, self.unlabel = split2PT(data)
        else:
            self.positive = positive_data
            self.unlabel = unlabel_data
        self.data = data
        self.prtrainreaderlength = len(self.positive)
        self.P_length = len(self.positive)
        self.U_length = len(self.unlabel)
        self.embeding_model = embeding_model

    def data_embedding(self, data):
        assert isinstance(data, pd.core.series.Series), "error type"
        if len(data) == 4: 
            smiles, sequence, interaction, id = data
            unimol_repr = self.embeding_model.smiles_embedding(smiles)
            protein = self.embeding_model.sequence_embedding(sequence=sequence,id=id)
            label = np.array(interaction, dtype=np.float32)
            label = torch.LongTensor(label)
            return (unimol_repr, protein, label)
        elif len(data) == 3:
            smiles, sequence, interaction = data
            unimol_repr = self.embeding_model.smiles_embedding(smiles)
            protein = self.embeding_model.sequence_embedding(sequence=sequence)
            label = np.array(interaction, dtype=np.float32)
            label = torch.LongTensor(label)
            return (unimol_repr, protein, label)
        else :
            print("error dataframe length")
        
    def __len__(self):
        return self.P_length+self.U_length

    def __getitem__(self, index):
        if self.data is not None:
            return self.data_embedding(self.data.iloc[index])
        if index < self.P_length:
            return self.data_embedding(self.positive.iloc[index])
        return self.data_embedding(self.unlabel.iloc[index-self.P_length])


class TrainReader(TestReader):
    """
    Only for train
    Need split positive and unlabel data
    """
    
    def __init__(self, embeding_model,data=None, positive_data=None, unlabel_data=None):
        """"
        U_m_P_savepath: to save data that may be positive in unlabel data 
        """
        if data is None:
            assert positive_data is not None and unlabel_data is not None,'wrong input'
        if data is not None: 
            assert positive_data is None and unlabel_data is None, 'wrong input'
        if data is not None:
            self.data = data
            self.positive, self.unlabel = split2PT(data)
        else:
            self.positive = positive_data
            self.unlabel = unlabel_data
        self.P_length = len(self.positive)
        self.U_length = len(self.unlabel)
        self.embeding_model = embeding_model
        self.ifpu = False
        
    def __len__(self):
        return self.P_length*2 if self.ifpu else self.P_length + self.U_length
    def __getitem__(self, index):
        if index < self.P_length:
            return self.data_embedding(self.positive.iloc[index])
        selfindex = index - self.P_length
        return self.data_embedding(self.unlabel.iloc[selfindex])

def get_reaction_pre_data(config):
    p_train = pd.read_csv(config.positive_train)
    n_train = pd.read_csv(config.negative_train)
    p_test = pd.read_csv(config.positive_test)
    n_test = pd.read_csv(config.negative_test)
    return p_test, n_test, p_train, n_train



