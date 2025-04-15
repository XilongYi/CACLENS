import torch
import torch.nn.functional as F

def info_message(message, *args, **kwargs):
    print(message.format(*args), **kwargs)

def get_cluster_center(model_emb, ec_id_dict):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for i, ec in enumerate(list(ec_id_dict.keys())):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
        message = 'Get cluster center {}/{}'
        info_message(message, i, len(list(ec_id_dict.keys())), end="\r")
    return cluster_center_model


def dist_map_helper_dot(keys1, lookup1, keys2, lookup2):
    dist = {}
    lookup1 = F.normalize(lookup1, dim=-1, p=2)
    lookup2 = F.normalize(lookup2, dim=-1, p=2)
    for i, key1 in enumerate(keys1):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm**2
        #dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    for i, key1 in enumerate(keys1):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def get_dist_map(ec_id_dict, em_emb, device, model=None, dot=False):
    '''
    Get the distance map for training, size of (N_EC_train, N_EC_train)
    between all possible pairs of EC cluster centers
    '''
    # inference all queries at once to get model embedding
    if model is not None:
        model_emb = model(em_emb.to(device=device))
        model_emb = model_emb.squeeze(1)
    else:
        # the first distance map before training comes from ESM
        model_emb = em_emb

    # calculate cluster center by averaging all embeddings in one EC
    cluster_center_model = get_cluster_center(model_emb, ec_id_dict)
    # organize cluster centers in a matrix
    total_ec_n, out_dim = len(ec_id_dict.keys()), model_emb.size(-1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device)
    # calculate pairwise distance map between total_ec_n * total_ec_n pairs
    print(f'Calculating distance map, number of unique EC is {total_ec_n}')
    if dot:
        model_dist = dist_map_helper_dot(ecs, model_lookup, ecs, model_lookup)
    else:
        model_dist = dist_map_helper(ecs, model_lookup, ecs, model_lookup)
    return model_dist


def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    ids = list(id_ec_test.keys())
    print(f'Calculating eval distance map, between {len(ids)} test ids '
          f'and {total_ec_n} train EC cluster centers')
    if dot:
        eval_dist = dist_map_helper_dot(ids, model_emb_test, ecs, model_lookup)
    else:
        eval_dist = dist_map_helper(ids, model_emb_test, ecs, model_lookup)
    return eval_dist


def get_random_nk_dist_map(emb_train, rand_nk_emb_train,
                           ec_id_dict_train, rand_nk_ids,
                           device, dot=False):
    '''
    Get the pair-wise distance map between 
    randomly chosen nk ids from training and all EC cluster centers 
    map is of size of (nk, N_EC_train)
    '''
    cluster_center_model = get_cluster_center(emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device)
    if dot:
        random_nk_dist_map = dist_map_helper_dot(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    else:
        random_nk_dist_map = dist_map_helper(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    return random_nk_dist_map
