# -*- coding: utf-8 -*-
# Author: Xilong Yi
# Copyright (c) WuLab, Inc. and its affiliates. All Rights Reserved

from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoModel, AutoTokenizer
from unimol_tools import UniMolRepr
from functools import lru_cache
import onnxruntime as ort
import h5py
import os
import re
import torch
import esm
import numpy as np
import pandas as pd
from .tools import info_message
current_file_path = os.path.abspath(__file__) if '__file__' in globals() else os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))


class EmbedingModel:
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  protein_model="t5", reaction_model="unimol" ) :
        self.project_root = os.getcwd()
        self.protein_model = protein_model
        self.reaction_model = reaction_model

        if self.protein_model=="esm36":
            self.esm36_model, self.batch_converter, self.clip_model = get_esm36_clip_model(device)
        elif self.protein_model=="t5":
            self.prot_t5_model, self.prot_t5_tokenizer,self.clip_model = get_t5_clip_model(device)
        else:
            raise ValueError(f"Unsupported protein model: {self.protein_model}. Expected 'esm36' or 't5'.")

        if self.reaction_model == "unimol":
            self.unimol_model = UniMolRepr(data_type='molecule', remove_hs=True)
        elif self.reaction_model == "molformer":

            model_path = "/root/CLIP_CROSS/model/mol"
            self.molformer_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
            self.molformer_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        else:
            raise ValueError(f"Unsupported reaction model: {self.reaction_model}. Expected 'unimol' or 'molformer.")
        

# @lru_cache
    @lru_cache(maxsize=None)
    def sequence_embedding(self, sequence=None, id=None, protein_model=None): 
        """
        protein sequence embedding
        """
        if protein_model is None:
            protein_model = self.protein_model
        if id:
            file_path = os.path.join(project_root, f"data/protein_embedding/{protein_model}/{id}.pt")
            if os.path.exists(file_path):
                try:
                    protein_embedding = torch.load(file_path)
                    return protein_embedding
                except Exception:
                    pass

        protein_embedding = self.generate_protein_em(sequence)
        if id:
            save_protein_em(protein_embedding, id, protein_model)
        
        return protein_embedding
    
    def generate_protein_em(self, sequence, protein_model=None):
        if protein_model is None:
            protein_model = self.protein_model

        if protein_model == "esm36":
            embedding = embed_sequences_esm36_clip(
                sequence, self.esm36_model, self.batch_converter, self.clip_model
            )
        else:
            embedding = embed_sequences_t5_clip(
                sequence, self.prot_t5_model, self.prot_t5_tokenizer, self.clip_model
            )
        return torch.FloatTensor(embedding).unsqueeze(0)
    
    def ec_id_dict_embedding(self, ec_id_dict, id_seq_dict, device, protein_em_model=None):
        '''
        Loading esm embedding in the sequence of EC numbers
        prepare for calculating cluster center by EC
        '''
        model_emb = []
        for ec in list(ec_id_dict.keys()):
            ids_for_query = list(ec_id_dict[ec])
            esm_to_cat = []
            for i, id in enumerate(ids_for_query):
                em = self.sequence_embedding(sequence=id_seq_dict[id], id=id)
                em = em.unsqueeze(0)
                esm_to_cat.append(em)
            model_emb = model_emb + esm_to_cat

            message = "ec_id_dict embedding {}/{}"
            info_message(message, i, len(list(ec_id_dict.keys())), end="\r")
        return torch.cat(model_emb).to(device=device)

    def id_ec_dict_embedding(self, id_ec_test, id_seq_dict_test, device):
        model_emb = []
        ids_for_query = list(id_ec_test.keys())
        for id in ids_for_query:
            em = self.sequence_embedding(sequence=id_seq_dict_test[id])
            em = em.unsqueeze(0)
            model_emb.append(em)
        model_emb = torch.cat(model_emb).to(device=device)
        model_emb = model_emb.squeeze(1)
        return model_emb

    # @lru_cache
    @lru_cache(maxsize=None)
    def smiles_embedding(self, smiles, reaction_model=None):
        """
        smiles embedding
        """
        if reaction_model is None:
            reaction_model = self.reaction_model

        reactants, products = smiles.split(">>")
        if reaction_model == "unimol":
            file_path = project_root+'/data/smiles_embedding/unimol.h5'
        else:
            file_path = project_root+'/data/smiles_embedding/molformer.h5'

        try:
            with h5py.File(file_path, 'r') as h5f:
                reactants_em = np.array(h5f[reactants][:])
                products_em = np.array(h5f[products][:])
            smiles_embedding = torch.FloatTensor(np.vstack((reactants_em, products_em)))
            # print(f"smiles_embedding{smiles_embedding.shape}")
            return smiles_embedding

        except Exception:
            reaction_em, reactants, products, reactants_em, products_em = self.generate_reaction_em(reactants, products, reaction_model)
            save_reaction_em(reactants, products, reactants_em, products_em, file_path)
        
            return reaction_em

    def generate_reaction_em(self, reactants, products, reaction_model):
        if reaction_model == "unimol":
            reactants_em = self.unimol_embedding(reactants)
            products_em = self.unimol_embedding(products)
        else:
            reactants_em = self.molformer_embedding(reactants)
            products_em = self.molformer_embedding(products)

        reaction_em = torch.FloatTensor(
            np.vstack((reactants_em, products_em)))
        return reaction_em, reactants, products, reactants_em, products_em
        
    def molformer_embedding(self, smiles):
        smiles_input = self.molformer_tokenizer(list(smiles), padding=True, return_tensors="pt")
        with torch.no_grad():
            output = self.molformer_model(**smiles_input)  
            em = output.pooler_output
            em = em.detach().cpu().numpy()
        return em

    def unimol_embedding(self, smiles):
        em = self.unimol_model.get_repr(smiles, return_atomic_reprs=True)['cls_repr']
        return em
    
def save_protein_em(protein_embedding, id, protein_model):

    file_path = os.path.join(project_root, f"data/protein_embedding/{protein_model}/{id}.pt")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(protein_embedding, file_path)
    except Exception as e:
        print(f"Failed to save embedding for id {id}: {e}")

def save_reaction_em(reactants, products, reactants_em, products_em, file_path):
    with h5py.File(file_path, 'a') as h5f:
        if reactants not in h5f:  
            h5f[reactants] = reactants_em #['cls_repr'] 
        if products not in h5f:
            h5f[products] = products_em#['cls_repr'] 
    
def embed_sequences_t5_clip(
        sequence, 
        model_t5, 
        tokenizer,
        clip_model,
        model_key="t5xl_half", 
        device=torch.device('cuda:0'), 
        gpu=0, 
        maxlen=1500,
        output="t5_clip"
    ):
    """
    Embed a single protein sequence and return the embedding vector.

    Parameters:
    - sequence (str): The protein sequence to embed.
    - model_key (str): The key for the model to use.
    - gpu (int): The GPU index to use.
    - maxlen (int): Maximum sequence length; longer sequences will be skipped.
    - output: str, output format, supports "t5_clip" and "t5"
    Returns:
    - np.ndarray: The embedding vector of the sequence.
    """
    assert model_t5 is not None, "Failed to load T5 model"
    assert tokenizer is not None, "Failed to load T5 tokenizer"
    assert clip_model is not None, "Failed to load CLIP model"
    with torch.no_grad():
        ids = tokenizer(
            sanitize_sequence(sequence),
              add_special_tokens=True, padding="max_length",
              truncation=True, max_length=maxlen)
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        embedding_repr = model_t5(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )

        mean_repr = embedding_repr.last_hidden_state[0, : len(sequence)].mean(dim=0)
        t5_out = mean_repr.detach().cpu().numpy()
        # print(f"t5_out.shape{t5_out.shape}")
        # Normalize the vector
        t5_out /= np.linalg.norm(t5_out)
        if output == "t5":
            return t5_out
        # Use the CLIP model for embedding
        x = clip_model.predict(t5_out)
    return x


def embed_sequences_esm36_clip(
        sequence, 
        esm36_model,
        batch_converter,
        clip_model,
        device=torch.device('cuda:0'), 
        maxlen=1500,
        output="esm_clip"
    ):
    """
    Embed a single protein sequence and return the embedding vector.

    Parameters:
    - sequence (str): The protein sequence to embed.
    - model: The preloaded ESM2 model (e.g., esm2_t36_3B_UR50D).
    - alphabet: The alphabet corresponding to the ESM2 model.
    - clip_model: The CLIP model for joint embedding.
    - device (torch.device): Device to use for computation (default is 'cuda:0').
    - maxlen (int): Maximum sequence length; longer sequences will be skipped.
    - output (str): Output format, supports "esm_clip" and "esm".

    Returns:
    - np.ndarray: The embedding vector of the sequence.
    """
    assert esm36_model is not None, "Failed to load ESM model"
    assert batch_converter is not None, "Failed to load ESM batch_converter"
    assert clip_model is not None, "Failed to load CLIP model"
    sequence = sequence[:maxlen]
    _, _, tokens = batch_converter([("", sequence.replace("*", "<mask>"))])
    with torch.no_grad():
        results = esm36_model(tokens.to(device), repr_layers=[36])
        token_representations = results["representations"][36]  # 获取第36层嵌入
    rep = token_representations[0, 1 : 1 + len(sequence)]
    mean_repr = rep.mean(dim=0) 
    esm_out = mean_repr.detach().cpu().numpy()
        # print(f"t5_out.shape{t5_out.shape}")
        # Normalize the vector
    esm_out /= np.linalg.norm(esm_out)
    # Use the CLIP model for embedding
    x = clip_model.predict(esm_out)
    return x

    

class ONNXModel:
    """Wrapper for an ONNX model to provide a more familiar interface."""

    def __init__(self, path):
        path = str(path)
        self.model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, x: np.ndarray, apply_norm: bool = True):
        assert x.ndim == 1
        if apply_norm:
            x /= np.linalg.norm(x)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self.model.run(None, {"input": x[None, :]})[0].squeeze()

def sanitize_sequence(sequence):
    """Replace uncommon amino acids with X and insert whitespace."""
    retval = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    return retval

def get_t5_clip_model(device=torch.device('cuda:0'), model_key="t5xl_half"):

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    if tokenizer is None:
        print("Tokenizer is None. Check the model path and tokenizer files.")
    model_t5.eval()
    model_t5.to(device)
    clip_model_path = project_root+"/model/proteinclip/proteinclip_prott5.onnx"
   
    clip_model = ONNXModel(clip_model_path)
    return model_t5, tokenizer,clip_model

def get_esm36_clip_model(device=torch.device('cuda:0')):

    esm36_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm36_model = esm36_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()  
    clip_model_path = project_root+"/model/proteinclip/proteinclip_esm2_36.onnx"
    
    clip_model = ONNXModel(clip_model_path)
    return esm36_model, batch_converter, clip_model
