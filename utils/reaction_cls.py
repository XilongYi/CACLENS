# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved
import pandas as pd
import tmap as tm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
import json
from torch.utils.data import Dataset
import random
from .tools import info_message
from sklearn.metrics.pairwise import cosine_similarity
scaler = GradScaler()

def reaction_data_embedding(embedding_model, data):
    data_em = []
    for i, r in enumerate(data["rxn"]):
        x = embedding_model.smiles_embedding(r)
        data_em.append(x)
        message = "Schneider50k embedding {}/{}"
        info_message(message, i, len(data["rxn"]), end="\r")
    with open('./data/rcls/rxnclass2id.json', 'r') as file:
        rxnclass2id = json.load(file)
    label = [rxnclass2id[c] for c in data["rxn_class"].values]
    
    return data_em, label

def get_reaction_cls_data(embedding_model, reaction_data):
    train = reaction_data[reaction_data['split'] == 'train']
    train_em,train_labels = reaction_data_embedding(embedding_model, train)
    test = reaction_data[reaction_data['split'] == 'test']
    test_em,test_labels = reaction_data_embedding(embedding_model, test)
    return train_em, train_labels, test_em, test_labels

def ec_reaction_data_embedding(embedding_model, data):
    data_em = []
    for i, r in enumerate(data["Reaction"]):
        x = embedding_model.smiles_embedding(r)
        data_em.append(x)
        message = "ec_reaction embedding {}/{}"
        info_message(message, i, len(data["Reaction"]), end="\r")
    label = data['EC number'].to_list()
    return data_em, label

def ec_get_reaction_cls_data(embedding_model, reaction_data):
    train = reaction_data[reaction_data['split'] == 'train']
    train_em,train_labels = ec_reaction_data_embedding(embedding_model, train)
    test = reaction_data[reaction_data['split'] == 'test']
    test_em,test_labels = ec_reaction_data_embedding(embedding_model, test)
    return train_em, train_labels, test_em, test_labels



class ContrastiveReactionDataset(Dataset):
    def __init__(self, data_em, labels, p_samples=5, n_samples=10):

        self.reactions = data_em
        self.labels = labels
        self.label2em_dict = self._create_label2em_dict()
        self.p_samples = p_samples
        self.n_samples = n_samples
        
    def _create_label2em_dict(self):
        label2em_dict = {}
        for idx, label in enumerate(self.labels):
            if label not in label2em_dict:
                label2em_dict[label] = []
            label2em_dict[label].append(idx)
        return label2em_dict

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        anchor_reaction = self.reactions[idx]
        anchor_label = self.labels[idx]
        positive_idxs = self._get_positive_samples(anchor_label, idx)
        positive_reactions = [self.reactions[i] for i in positive_idxs]
        negative_idxs = self._get_negative_samples(anchor_label)
        negative_reactions = [self.reactions[i] for i in negative_idxs]
        all_reactions = [anchor_reaction] + positive_reactions + negative_reactions
        all_reactions = torch.stack(all_reactions, dim=0)

        return all_reactions#, anchor_label

    def _get_positive_samples(self, anchor_label, anchor_idx):
        positive_idxs = [idx for idx in self.label2em_dict[anchor_label] if idx != anchor_idx]
        return random.sample(positive_idxs, self.p_samples)

    def _get_negative_samples(self, anchor_label):
        all_labels = list(self.label2em_dict.keys())
        all_labels.remove(anchor_label)  
        all_negative_idxs = []
        for label in all_labels:
            all_negative_idxs.extend(self.label2em_dict[label])  
        negative_samples = random.sample(all_negative_idxs, self.n_samples)
        return negative_samples
    

class EC_ContrastiveReactionDataset(Dataset):
    def __init__(self, data_em, labels, p_samples=9, n_samples=30):
        self.reactions = data_em
        self.labels = labels
        self.label2em_dict = self._create_label2em_dict()
        self.p_samples = p_samples
        self.n_samples = n_samples

    def _create_label2em_dict(self):
        label_dict = {}
        for reaction, label in zip(self.reactions, self.labels):
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(reaction)
        return label_dict

    def _get_positive_samples(self, anchor_label, exclude_reaction):
        def ec_level_match(label1, label2, level):
            return ".".join(label1.split('.')[:level]) == ".".join(label2.split('.')[:level])

        levels = [4, 3, 2, 1]  
        positives = []
        for level in levels:
            for label, reactions in self.label2em_dict.items():
                if ec_level_match(label, anchor_label, level):
                    filtered = [r for r in reactions if not torch.equal(r, exclude_reaction)]
                    positives.extend(filtered)
            positives = list(set(positives))
            if len(positives) >= self.p_samples:
                break
        return random.sample(positives, self.p_samples)

    def _get_negative_samples(self, anchor_label, exclude_reaction):
        anchor_first_digit = anchor_label.split('.')[0]
        negatives = []
        for label, reactions in self.label2em_dict.items():
            if label.split('.')[0] != anchor_first_digit:
                filtered = [r for r in reactions if not torch.equal(r, exclude_reaction)]
                negatives.extend(filtered)
                negatives = list(set(negatives))
            if len(negatives) >= self.n_samples:
                break
        return random.sample(negatives, self.n_samples)

    def __len__(self):
        return len(self.reactions)
    def __getitem__(self, idx):
        anchor_reaction = self.reactions[idx]
        anchor_label = self.labels[idx]
        positive_reactions = self._get_positive_samples(anchor_label, anchor_reaction)
        negative_reactions = self._get_negative_samples(anchor_label, anchor_reaction)
        all_reactions = [anchor_reaction] + positive_reactions + negative_reactions
        all_reactions = torch.stack(all_reactions, dim=0)

        return all_reactions


def get_rcls_dataloader(dataset, batch_size):
    params = {
        'batch_size': batch_size,
        'shuffle': True,
    }
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    return dataloader
    
def rcls_train_epoch(model, train_loader, rcls_criterion, optimizer, temp, n_pos):
    model.train()
    running_loss = 0.0

    for i, batch_embeddings in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = batch_embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with autocast():
            outputs = model(inputs)
            loss = rcls_criterion(outputs, temp, n_pos)
        scaler.scale(loss).backward() 
        scaler.step(optimizer)        
        scaler.update()  
        running_loss += loss.item()
        message = "Reaction Classification Tower: Training Progress {}/{}, loss: {:.5f}"
        info_message(message, i, len(train_loader), running_loss/(i + 1), end="\r")
    train_loss = running_loss
   
    return train_loss/(i + 1)

def reaction_ec_search(reaction, embedding_model, rcls_tower, ec_num=5, ec_level=3):
    schneider_df = pd.read_csv('./data/rcls/ec_reaction_data_split.csv')
    new_row = {
        'Reaction': reaction,
        'EC number': 'query',
        'split': 'query'
    }

    schneider_df = pd.concat([schneider_df, pd.DataFrame([new_row])], ignore_index=True)
    print(schneider_df.shape)
    fps = np.empty((0, rcls_tower.out_dim))
    with torch.no_grad():
        x = embedding_model.smiles_embedding(reaction).unsqueeze(0)
        out =  rcls_tower(x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        fps = np.array(out.detach().cpu().numpy()) 
        fps = fps.squeeze(1)

        
    caclens_fps = np.load('./data/rcls/reaction_ec_for_search.npz')['fps']
    caclens_fps = np.vstack((caclens_fps, fps))

    
    schneider_df['fp'] = [fp for fp in caclens_fps]
    train_df = schneider_df[schneider_df['split'] != 'query']
    train_fps = [np.array(fp.tolist()) for fp in train_df['fp'].values]
    query_fp = schneider_df.iloc[-1]['fp']
    print(f'Query Reaction: {reaction}')
    similarities = cosine_similarity([query_fp], train_fps)[0]
    top_n_indices = similarities.argsort()[-ec_num:][::-1]
  
    pred_labels = [('.'.join(train_df.iloc[i]['EC number'].split('.')[:ec_level]) + '.*') for i in top_n_indices]

    return pred_labels
