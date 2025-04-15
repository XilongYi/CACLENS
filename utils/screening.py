import argparse
import pandas as pd
import os
import re
from .contrastive.infer import infer_maxsep
from .reaction_feasibility_pre import PredictReader, rpre_predict
from torch.utils.data import DataLoader
from .tools import compute_distance
from utils.embedding import EmbedingModel



def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_dirs(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        os.remove(file_path)

def count_matching_segments(ec_fr_list, ec_number):
    total_count = 0
    for ec_fr in ec_fr_list:
        segments_pattern = ec_fr.split('.')
        segments_number = ec_number.split('.')
        count = 0
        for p_seg, n_seg in zip(segments_pattern, segments_number):
            if p_seg == "*" or p_seg == n_seg:
                count += 1
            else:
                break
        total_count += count
    return total_count

def filter_by_ec_number(ec_fr, model_input, embedding_ef, model, enzymes_to_keep=0.7):
    ensure_dirs("./candidate_enzymes/cache")
    remove_dirs("./candidate_enzymes/cache")
    ensure_dirs("./results")
    


    result_data = []
    ec_input = model_input.copy()
    ec_input = ec_input.rename(columns={'Protein': 'Sequence','ID': 'Entry'})
    ec_input[['EC number']] = '0.0.0.0'
    ec_input = ec_input[['Entry', 'EC number', 'Sequence']]
    ec_input.to_csv('./candidate_enzymes/cache/new_input.csv', index=False, sep='\t')

    infer_maxsep('split100', 'new_input', model, embedding_ef, report_metrics=False)

    ec_result = pd.read_csv('./candidate_enzymes/cache/ec_maxsep.csv', header=None, names=["ID", "EC"])
    for _, row in ec_result.iterrows():
        protein_id = row["ID"]
        ec_predictions = row["EC"].split(";")  
        match_count = 0
        for ec_number in ec_predictions:  
            count = count_matching_segments(ec_fr, ec_number)
            match_count += count
        result_data.append({"ID": protein_id, "MatchCount": match_count})
    result_df = pd.DataFrame(result_data)
    df_sorted = result_df.sort_values(by="MatchCount", ascending=False)

    if len(df_sorted) >= 10:
        num_rows_to_keep = int(len(df_sorted) * enzymes_to_keep)
        result_filtered = df_sorted.head(num_rows_to_keep)
        result_with_seq = pd.merge(result_filtered, model_input, on='ID', how='left')
        result = result_with_seq[['ID', 'Reaction', 'Protein']]
    else:
        result_with_seq = pd.merge(df_sorted, model_input, on='ID', how='left')
        result = result_with_seq[['ID', 'Reaction', 'Protein']]

    del embedding_ef
    import gc
    gc.collect()
    remove_dirs("./candidate_enzymes/cache")
    return result

def get_enzymes_df(enzyme_in):
    if enzyme_in.endswith(".csv"):
        data = pd.read_csv('./candidate_enzymes/' + enzyme_in)
        columns = data.columns.tolist()
        if len(columns) == 1:
            assert columns[0] == 'Protein', f"Invalid DataFrame: Expected single column 'Protein', but got {columns}"
            data['ID'] = [f'Protein_{i}' for i in range(len(data))]
        elif len(columns) == 2:
            assert set(columns) == {'Protein', 'ID'}, f"Invalid DataFrame: Expected columns 'Protein' and 'ID', but got {columns}"
            new_ids = []
            counts = {}
            for id_val in data['ID']:
                counts[id_val] = counts.get(id_val, 0) + 1
                if counts[id_val] == 1:
                    new_ids.append(id_val)
                else:
                    new_ids.append(f"{id_val}_{counts[id_val]}")
            data['ID'] = new_ids
        else:
            raise ValueError(f"Invalid DataFrame: Expected 1 or 2 columns, but got {len(columns)} columns: {columns}")
    else:
        data = pd.DataFrame({'ID': ['Protein_0'],'Protein': [enzyme_in]})

    return data

def reaction_feasibility_prediction(data, embedding_model, rpre_tower):
    predictdata = PredictReader(embedding_model, data)
    predict_dataloader = DataLoader(predictdata, batch_size=512)
    Y, S = rpre_predict(rpre_tower, predict_dataloader)
    data.insert(loc=2,column="Label", value=Y)
    data.insert(loc=3,column="Score", value=S)

    return data



