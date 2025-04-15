# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

import argparse
import pandas as pd
import os
import re
from utils.screening import *
from utils.utils import *
from utils.reaction_cls import reaction_ec_search
from model.model import ReactionClassificationTower,ECPredictionTower,ReactionPredictionTower
from utils.embedding import EmbedingModel
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.cgc_framework import PLE_Protein,PLE_Reaction
from config import CFG
import sys
# cfg
seed_everything(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding_combinations = {
"EF": ("esm36", "molformer"),
"EU": ("esm36", "unimol"),
"TF": ("t5", "molformer"),
"TU": ("t5", "unimol")
}
class EC_Config:
    def __init__(self):
        self.hid_dim = 512
        self.out_dim = 256

class RCls_Config:
    def __init__(self):
        self.reaction_model_name = "molformer"
        self.hidden_dim=256
        self.out_dim=256

class RPre_Config:
    def __init__(self, reaction_model_name):
        self.layer_num = 2
        self.num_heads = 8
        self.hid_dim = 128
        self.norm_shape = 128
        self.reaction_model_name = reaction_model_name

def get_predict_model(m_type):
    protein_model_name, reaction_model_name = encoding_combinations[m_type]
    print(f"Protein Embedding Model: {protein_model_name}; Reaction Embedding Model: {reaction_model_name}")
    ec_config = EC_Config()
    rcls_config = RCls_Config()
    rpre_config = RPre_Config(reaction_model_name)
    ## model     
    ple_protein = PLE_Protein().to(device)
    ple_reaction = PLE_Reaction().to(device)

    ec_tower = ECPredictionTower(
        ec_config.hid_dim, ec_config.out_dim, ple_protein=ple_protein).to(device)
    ec_tower.load_state_dict(torch.load("./model/model_save/EF/ec_tower.pth"))

    rcls_tower = ReactionClassificationTower(rcls_config, ple_reaction=ple_reaction).to(device)
    rcls_tower.load_state_dict(torch.load("./model/model_save/EF/rcls_tower_ec_molformer.pth"))

    rpre_tower = ReactionPredictionTower(
        rpre_config, ple_protein=ple_protein, ple_reaction=ple_reaction).to(device)
    rpre_tower.load_state_dict(torch.load(f"./model/model_save/{m_type}/rpre_tower.pth"))

    return ec_tower, rcls_tower, rpre_tower

def get_embedding_model(m_type):

    protein_model_name, reaction_model_name = encoding_combinations[m_type]
    embedding_model = EmbedingModel(
        protein_model=protein_model_name, reaction_model=reaction_model_name)
    return embedding_model



def predict(smirks, model_input, m_type=None, f_method=None, ec_input=None):

    print(f"Selected model: {m_type}")

    model_input[['Reaction']] = smirks

    ec_tower, rcls_tower, rpre_tower = get_predict_model(m_type)

    if f_method == 'direct':
        embedding_model = get_embedding_model(m_type)
        model_out = reaction_feasibility_prediction(model_input, embedding_model, rpre_tower)

    elif f_method == 'ec':
        assert ec_input is not None, "ec_input should not be None!"
        embedding_ef = EmbedingModel(
        protein_model="esm36", reaction_model="molformer")
        input_ec_f = filter_by_ec_number(list(ec_input), model_input, embedding_ef, ec_tower)
        embedding_model = get_embedding_model(m_type)
        model_out = reaction_feasibility_prediction(input_ec_f, embedding_model, rpre_tower)

    elif f_method == 're2ec':
        embedding_ef = EmbedingModel(
        protein_model="esm36", reaction_model="molformer")
        ec_fr = reaction_ec_search(smirks, embedding_ef, rcls_tower)
        input_ec_f = filter_by_ec_number(ec_fr, model_input, embedding_ef, ec_tower)
        embedding_model = get_embedding_model(m_type)
        model_out = reaction_feasibility_prediction(input_ec_f, embedding_model, rpre_tower)

    else:
        raise Exception("Error: Value does not meet any conditions!")
    merged_df = pd.merge(model_input, model_out[['ID','Label','Score']], on="ID", how="left")
    merged_df.fillna(0, inplace=True)
    merged_df.to_csv(f'./results/model_out_{m_type}.csv',index=False)
    
    


sys.argv = [
    "predict.py",
    "--reactant", "COc1cc(C=CC(=O)[O-])cc(OC)c1O",
    "--product", "COc1cc(C=CC(=O)OCC[N+](C)(C)C)cc(OC)c1O",
    "--seq", "example.csv",
    "--screening_type", 're2ec',
    "--model", "TF",
    "--ec_num", "3.1.1.*"
]



def main():
    parser = argparse.ArgumentParser(description="Enzyme screening based on reaction and EC number.")
    
    # Define command-line arguments
    parser.add_argument("--reactant", type=str, required=True, 
                        default="COc1cc(C=CC(=O)[O-])cc(OC)c1O",
                        help="Input reactant (chemical formula, SMILES, etc.)")
    
    parser.add_argument("--product", type=str, required=True,
                        default="COc1cc(C=CC(=O)OCC[N+](C)(C)C)cc(OC)c1O", 
                        help="Input product (chemical formula, SMILES, etc.)")
    
    parser.add_argument("--seq", type=str, required=True, 
                        default="example.csv",
                        help="Enzyme sequence (CSV file path or direct input)")
    
    parser.add_argument("--screening_type", type=str, choices=['re2ec', 'ec', 'direct'],
                        default="direct", 
                        help="Screening type (re2ec, ec, or direct)")
    
    parser.add_argument("--model", type=str, choices=['EU', 'EF', 'TU', 'TF'],
                        default="TF", 
                        help="Encoding model type: 'EU' (ESM2 + UniMol), 'EF' (ESM2 + MolFormer),"
                        " 'TU' (ProtT5 + UniMol), 'TF' (ProtT5 + MolFormer).")
    parser.add_argument("--ec_num", type=str, default='3.1.1.*', help="EC number for filtering (optional)")
    args = parser.parse_args()

    smirks = args.reactant+ ">>" +args.product
    model_input = get_enzymes_df(args.seq)
    
    predict(smirks, model_input, m_type=args.model, f_method=args.screening_type, ec_input=args.ec_num)
    
if __name__ == '__main__':
    main()