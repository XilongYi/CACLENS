import torch
from .distance_map import *
from .evaluate import *
import pandas as pd
import warnings
import pickle

import os
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def infer_maxsep(train_data, test_data, model, embedding_model, report_metrics = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/ec_pre/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./candidate_enzymes/cache/' + test_data + '.csv')
    model.eval()

    with open(f'./data/distance_map/{train_data}_em.pkl', 'rb') as file:
        data = pickle.load(file)
    emb_train = model(data)
    emb_train = emb_train.squeeze(1)
    data = pd.read_csv('./candidate_enzymes/cache/' + test_data + '.csv', sep='\t')
    id_seq_dict_test = dict(zip(data['Entry'], data['Sequence']))
    data_test = embedding_model.id_ec_dict_embedding(id_ec_test, id_seq_dict_test, device)

    emb_test = model(data_test)
    emb_test = emb_test.squeeze(1)

    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    eval_df = pd.DataFrame.from_dict(eval_dist)

    out_filename = './candidate_enzymes/cache/ec'
    write_max_sep_choices(eval_df, out_filename)


