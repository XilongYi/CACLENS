# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn.functional as F
from torch import nn
import math
import os

class FFN(nn.Module):
    """
    Two linear layers, Feed Foward
    """
    def __init__(self, ffn_input_num, ffn_hidden_num, ffn_outputs_num, **kwargs):
        """
        ffn_input_num: hid_dim
        ffn_hidden_num: hid_dim
        ffn_outputs_num: hid_dim
        """
        super().__init__()
        self.linear1 = nn.Linear(ffn_input_num, ffn_hidden_num)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.linear2 = nn.Linear(ffn_hidden_num, ffn_outputs_num)
        if 'drop_rate' in kwargs:
            self.dropout = nn.Dropout(kwargs['drop_rate'])
        else:
            self.dropout = nn.Dropout(0.1)
    def forward(self, X):
        return self.linear2(self.dropout(self.activation(self.linear1(X))))

class AddNorm(nn.Module):
    ''' dropout -> add X -> normalization'''
    def __init__(self, norm_shape, drop_rate, **kwargs):
        """
        norm_shape: layernorm parameter (64)
        """
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(norm_shape)
    def forward(self, X, y):
        return self.ln(self.dropout(y)+X)



        
class ProteinExpert(nn.Module):
    def __init__(self, hid_dim=128):
        super(ProteinExpert, self).__init__()
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        #self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        
        return x
    
class ReactionExpert(nn.Module):
    def __init__(self):
        super(ReactionExpert, self).__init__()
        self.fc1 = nn.Linear(512,512)
        #self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(512,512)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        
        return x

class ReactionChangeShape(nn.Module):
    def __init__(self):
        super(ReactionChangeShape, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (2, 3), padding='same')
        self.conv2 = nn.Conv2d(64, 32, (1, 3), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

    def change_shape(self,tensor):
        """
        Reshape the dimensions from (dim1, dim2, dim3) to (dim1 * dim2, dim3)
        """
        batch_size, dim1, dim2, dim3 = tensor.shape
        tensor = tensor.permute(0, 1, 3, 2)
        tensor = tensor.contiguous()
        output_tensor = tensor.view(batch_size, dim1 * dim2, dim3)
        output_tensor
        return output_tensor
    
    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.pool(F.leaky_relu(self.conv1(x))) 
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.change_shape(x)
        return x  #(b,64,128)

class ProteinChangeShape(nn.Module):
    def __init__(self, hid_dim=None):
        super(ProteinChangeShape, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(64)
        self.fc = nn.Linear(128, hid_dim)

    def forward(self, x):
        
        # 输入形状 (batch, 1, 128)
        x = self.conv(x)         # 卷积，输出形状 (batch, 64, 128)
        x = self.batch_norm(x)   # 批归一化
        x = F.leaky_relu(x) 
        x = self.fc(x)        # ReLU 激活
        return x
    
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def transpose_qkv(X, num_heads):
    """
    For MultiHeadAttention (n heads to 1 large head):
    num_heads: The number of attention heads.
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

class MultiHeadAttentionBlock(nn.Module):
    """
    Softmax((q*k**T/dk**0.5)*v)
    """
    def __init__(self, hid_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.W_q = nn.Linear(hid_dim, hid_dim)
        self.W_k = nn.Linear(hid_dim, hid_dim)
        self.W_v = nn.Linear(hid_dim, hid_dim)
        self.W_o = nn.Linear(hid_dim, hid_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x1, x2):
        # Compute Query, Key, and Value
        q = transpose_qkv(self.W_q(x1), self.num_heads)  
        k = transpose_qkv(self.W_k(x2), self.num_heads)    
        v = transpose_qkv(self.W_v(x2), self.num_heads)
        # d = q.shape[-1]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])  
        attn_weights = self.softmax(attn_weights)
        # Compute attention output
        output = torch.bmm(attn_weights, v)  
        attn_output = transpose_output(output, self.num_heads)  
        return self.W_o(attn_output)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hid_dim, norm_shape, layer_num, num_heads, drop_rate=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        # Create multiple layers of CrossAttention
        self.layers = nn.ModuleList([self._build_cross_attention_layer(hid_dim, norm_shape, drop_rate) 
                                     for _ in range(layer_num)])

    def _build_cross_attention_layer(self, hid_dim, norm_shape, drop_rate):
        """
        Create multiple layers of CrossAttention
        """
        return nn.ModuleDict({
            'cross_attention1': MultiHeadAttentionBlock(hid_dim,self.num_heads),  # Attention mechanism for x1 -> x2
            'cross_attention2': MultiHeadAttentionBlock(hid_dim,self.num_heads),  # Attention mechanism for x2 -> x1 
            'ffn1': FFN(hid_dim, hid_dim, hid_dim),
            'ffn2': FFN(hid_dim, hid_dim, hid_dim),
            'addnorm1': AddNorm(norm_shape, drop_rate),
            'addnorm2': AddNorm(norm_shape, drop_rate)
        })

    def forward(self, x1, x2):
        for layer in self.layers:

            cross_attention1 = layer['cross_attention1']
            cross_attention2 = layer['cross_attention2']
            ffn1 = layer['ffn1']
            ffn2 = layer['ffn2']
            addnorm1 = layer['addnorm1']
            addnorm2 = layer['addnorm2']
            
            attn_output1 = cross_attention1(x1, x2)  
            x1 = addnorm1(attn_output1, ffn1(attn_output1))
            attn_output2 = cross_attention2(x2, x1)  
            x2 = addnorm2(attn_output2, ffn2(attn_output2))
            
        return x1, x2

class ReactionPredictionTower(nn.Module):
    def __init__(self, config, ple_protein=None, ple_reaction=None, drop_rate=0.1):
        super(ReactionPredictionTower, self).__init__()
        self.num_heads = config.num_heads
        self.ple_protein = ple_protein
        self.ple_reaction = ple_reaction
        if self.ple_reaction is None:
            self.reactionmodel = ReactionExpert()

        self.protein_cs = ProteinChangeShape(hid_dim=config.hid_dim)
        self.reaction_cs = ReactionChangeShape()
        self.multi_layer_cross_attention = MultiHeadCrossAttention(
            config.hid_dim, config.norm_shape, config.layer_num, self.num_heads, drop_rate)
        self.reaction_model_name = config.reaction_model_name
        if  self.reaction_model_name == "molformer":
            self.fc0 = nn.Linear(768, 512)
        self.dense = nn.Linear(config.hid_dim * 2, 128)
        self.dense2 = nn.Linear(128, 2)
        self.drop_final = nn.Dropout(0.1)
        
    def forward(self, x1, x2):
        # x1: smiles,x2: seq
        if self.reaction_model_name == "molformer":
            x1 = self.fc0(x1)
        if self.ple_protein and self.ple_reaction is not None:
            x1_list = self.ple_reaction(rpre_x=x1)
            x1 = x1_list[1]
            x2_list = self.ple_protein(rpre_x=x2)
            x2 = x2_list[1]
        else:
            x1 = self.reactionmodel(x1)
            
        x1 = self.reaction_cs(x1)
        x2 = self.protein_cs(x2)

        x1, x2 = self.multi_layer_cross_attention(x1, x2)
        # Concatenate the outputs of the two cross-attention layers
        combined_output = torch.cat((x1.mean(dim=1), x2.mean(dim=1)), dim=-1)  
        label = self.drop_final(F.leaky_relu(self.dense(combined_output)))
        logits = self.dense2(label) 
        return logits
    

    
class ECPredictionTower(nn.Module):
    def __init__(self, hid_dim, out_dim, ple_protein=None, drop_out=0.1):
        super(ECPredictionTower, self).__init__()
        self.ple_protein = ple_protein
        if self.ple_protein is None:
            self.proteinmodel = ProteinExpert()

        self.hidden_dim1 = hid_dim
        self.out_dim = out_dim
        self.drop_out = drop_out

        self.fc1 = nn.Linear(128, hid_dim)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        if self.ple_protein is not None:
            x_list = self.ple_protein(x)
            x = x_list[0]
        else:
            x = self.proteinmodel(ec_x=x)

        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        

        return x

class ReactionClassificationTower(nn.Module):
    def __init__(self, config, ple_reaction=None, hidden_dim=256, out_dim=256, drop_out=0.1):
        super(ReactionClassificationTower, self).__init__()
        self.reaction_model_name = config.reaction_model_name
        if self.reaction_model_name == "molformer":
            self.fc0 = nn.Linear(768, 512)

        self.ple_reaction = ple_reaction
        # if self.ple_reaction is None:
        #     self.reactionmodel = ReactionExpert()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.fc1 = nn.Linear(1024, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        if self.reaction_model_name == "molformer":
            x = self.fc0(x)
        
        if self.ple_reaction is not None:
            x_list = self.ple_reaction(cls_x=x)
            x = x_list[0]
        # else:
        #     x = self.reactionmodel(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch_size, n, _, _ = x.shape
        x = x.view(batch_size * n, 1024)
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)

        x = x.view(batch_size, n, -1)
        return x
        

    