import csv

from .distance_map import *
from .evaluate import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score
import numpy as np


def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i


def write_max_sep_choices(df, csv_name, first_grad=True, use_max_grad=False):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.columns:
 
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            all_test_EC.add(EC_i)
            ec.append(str(EC_i))
        ec_string = ';'.join(ec)
        csvwriter.writerow([col, ec_string])
    return


def write_pvalue_choices(df, csv_name, random_nk_dist_map, p_value=1e-5):
    out_file = open(csv_name + '_pvalue.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = p_value*nk
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold) or (i == 0):
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            else:
                break
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def write_random_nk_choices_prc(df, csv_name, random_nk_dist_map, p_value=1e-4, 
                                upper_bound=0.0025, steps=24):
    out_file = open(csv_name + '_randnk.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = np.linspace(p_value, upper_bound, steps)*nk
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold[-1]) or (i == 0):
                if i != 0:
                    dist_str = str(np.searchsorted(threshold, rank))
                else:
                    dist_str = str(0)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            else:
                break
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def write_top_choices(df, csv_name, top=30):
    out_file = open(csv_name + '_top' + str(top)+'.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    dists = []
    for col in df.columns:
        ec = []
        dist_lst = []
        smallest_10_dist_df = df[col].nsmallest(top)
        for i in range(top):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_str = "{:.4f}".format(dist_i)
            dist_lst.append(dist_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        dists.append(dist_lst)
        csvwriter.writerow(ec)
    return dists


def random_nk_model(id_ec_train, ec_id_dict_train, emb_train, n=10, weighted=False):
    ids = list(id_ec_train.keys())
    nk = n * 1000
    if weighted:
        P = []
        for id in id_ec_train.keys():
            ecs_id = id_ec_train[id]
            ec_densities = [len(ec_id_dict_train[ec]) for ec in ecs_id]
            # the prob of calling this id is inversely prop to 1/max(density)
            P.append(1/np.max(ec_densities))
        P = P/np.sum(P)
        random_nk_id = np.random.choice(
            range(len(ids)), nk, replace=True, p=P)
    else:
        random_nk_id = np.random.choice(range(len(ids)), nk, replace=False)

    random_nk_id = np.sort(random_nk_id)
    chosen_ids = [ids[i] for i in random_nk_id]
    chosen_emb_train = emb_train[random_nk_id]
    return chosen_ids, chosen_emb_train


def update_dist_dict_blast(emb_test, emb_train, dist, start, end,
                           id_ec_test, id_ec_train):

    id_tests = list(id_ec_test.keys())
    id_trains = list(id_ec_train.keys())
    dist_matrix = torch.cdist(emb_test[start:end], emb_train)
    for i, id_test in enumerate(id_tests[start:end]):
        dist[id_test] = {}
        # continue adding EC/dist pairs until have 20 EC
        idx_train_closest_sorted = torch.argsort(dist_matrix[i], dim=-1)
        count = 0
        while len(dist[id_test]) <= 10:
            idx_train_closest = idx_train_closest_sorted[count]
            dist_train_closest = dist_matrix[i][idx_train_closest].cpu().item()
            count += 1
            id_train_closest = id_trains[idx_train_closest]
            ECs_train_closest = id_ec_train[id_train_closest]
            for EC in ECs_train_closest:
                # if EC is not added to the dict
                if EC not in dist[id_test]:
                    # add EC/dist pair
                    dist[id_test][EC] = dist_train_closest
    return dist


def get_pred_labels_prc(out_filename, cutoff, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            if int(pred_ec_dist.split(":")[1].split("/")[1]) <= cutoff:
                preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label


def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
        
    return label_pos_dict

