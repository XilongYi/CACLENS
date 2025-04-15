# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

protein_em_model = "t5"
reaction_em_model = "molformer"
class CFG:
    def __init__(self):
        self.ec = EC_Config()
        self.r_cls = RCls_Config()
        self.r_pre = RPre_Config()
        self.epochs_1 = 600
        self.epochs_2 = 400
        self.protein_em_model = protein_em_model
        self.reaction_em_model = reaction_em_model
        self.weight_decay = 1e-9

class EC_Config:
    def __init__(self):
        self.name = "EC Prediction Config"
        self.adaptive_rate = 60
        self.batch_size = 512
        self.protein_em_model = protein_em_model
        self.hid_dim = 512
        self.out_dim = 256
        self.learning_rate = 1e-3  
        self.train_data = "./data/ec_pre/split100.csv"
        self.test_data = "./data/ec_pre/new.csv"
        self.n_pos = 9
        self.n_neg = 30
        self.temp = 0.1


class RCls_Config:
    def __init__(self):
        self.name = "Reaction Classification Config"
        self.batch_size = 512
        self.p_samples=9
        self.n_samples=30
        self.temp = 0.1
        self.data = "./data/rcls/schneider50k_28.csv"
        self.reaction_em_model = reaction_em_model
        self.learning_rate = 2.5e-4   

class RPre_Config:
    def __init__(self):
        self.name = "Reaction Prediction Config"
        self.batch_size = 512
        self.learning_rate = 1.5e-7  
        self.layer_num = 2
        self.num_heads = 8
        self.dropout = 0.1
        self.hid_dim = 128
        self.norm_shape = 128
        self.reaction_em_model = reaction_em_model
        self.state_dict_path = None
        self.protein_dim = 128
        self.ifpu = False
        self.ifsmoothing = True
        self.del_threshold = 0.85
        self.positive_train = "./data/rpre/pospairs_train_sample.csv"
        self.negative_train = "./data/rpre/negpairs_train_sample.csv"
        self.positive_test = "./data/rpre/pospairs_test.csv"
        self.negative_test = "./data/rpre/negpairs_test.csv"
        self.result_file_path = './pairs_log/'
        self.best_model_savepath = "./model/model_save"
        self.quantile = 0.9

