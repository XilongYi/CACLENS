# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

import os
import pandas as pd
import json
import random
import numpy as np
import csv
import pickle
from .contrastive.distance_map import get_dist_map
import torch


def get_ec_id_dict(csv_name: str):
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_distance(embedding_model, train_file):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict('./data/ec_pre/' + train_file + '.csv')
    data = pd.read_csv('./data/ec_pre/' + train_file + '.csv', sep='\t')
    id_seq_dict = dict(zip(data['Entry'], data['Sequence']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em_emb = embedding_model.ec_id_dict_embedding(ec_id_dict, id_seq_dict, device)
    esm_dist = get_dist_map(ec_id_dict, em_emb, device)
    pickle.dump(esm_dist, open('./data/distance_map/' + train_file + '.pkl', 'wb'))
    pickle.dump(em_emb, open('./data/distance_map/' + train_file + '_em.pkl', 'wb'))

def d2D(data, device=torch.device('cuda:0')):
    """
    data to device: only for torch.tensor!!!
    device default:torch.device('cuda:0')
    """
    if isinstance (data,(tuple, list)):
        return (d.to(device) for d in data)
    return data.to(device)

def split2PT(df:pd.DataFrame):
    """
    Make sure the order of the columns is [Compound, protein, label(0/1)]
    """
    return df[df.iloc[:,2] > 0.5].reset_index(drop=True), df[df.iloc[:,2] < 0.5].reset_index(drop=True)
def info_message(message, *args, **kwargs):
    print(message.format(*args), **kwargs)
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



