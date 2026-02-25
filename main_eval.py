import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random

import pytorch_lightning as pl


from models.CNN import CNN_LTE, norm_embeddings
from utils.tools import compute_init_periods
from data.Aalto.preprocessing import extract_features_classic, get_session_fixed_length, compute_feature_quantiles
# from main.data.preprocessing import get_session_fixed_length, augment_session, compute_feature_quantiles,
import conf

sequence_length = 128
embedding_size = 512
num_samples = 4

def extract_keystroke_features(session_key, sequence_length, start_idx=0):
    numeric = extract_features_classic(get_session_fixed_length(session_key, sequence_length, start_zero=False, start_idx=start_idx))
    return numeric

def get_session_fixed_length(session, sequence_length, start_zero=True, start_idx=0):
    num_tile = max(1, (sequence_length * 2) // session.shape[0]) + 2
    tiled = np.tile(session, (num_tile, 1))
    if start_zero:
        start_idx = 0
    else:
        start_idx = start_idx
    return tiled[start_idx:start_idx+sequence_length]

def prepare_data(session):
    numeric_list, key_list = [], []
    step = sequence_length // num_samples
    for start_idx in range(0, sequence_length, step):
        result = extract_keystroke_features(session, sequence_length=sequence_length, start_idx=start_idx)
        numeric, keys = result
        key_list.append(keys)

    numeric_array = np.array(numeric_list)
    key_array = np.array(key_list)

    return (numeric_array, key_array)

def get_embeddings(model, sessions):
    numeric_batches, key_batches = [], []

    for cur_sess in sessions:
        numeric, keys = prepare_data(data[cur_sess])
        numeric_batches.append(numeric)

    numeric_batches = np.array(numeric_batches)  # (batch, num_samples, seq_len, numeric_dim)
    batch_size, num_samples, seq_len, numeric_dim = numeric_batches.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numeric_input = torch.from_numpy(
        numeric_batches.reshape(batch_size * num_samples, seq_len, numeric_dim).astype(np.float32)
    ).to(device)

    embeddings = model(numeric_input)  # model expects only numeric

    embeddings = embeddings.cpu().reshape(batch_size, num_samples, -1).mean(dim=1)
    return embeddings

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = {k.replace("model.", ""): v
                  for k, v in checkpoint['state_dict'].items()
                  if not k.startswith("loss_function")}
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

def compute_distances(embeddings, comps, metric="euclidean"):
    distances = {}
    for comp in comps:
        e1, e2 = norm_embeddings(embeddings[comp[0]]), norm_embeddings(embeddings[comp[1]])
        if metric == "euclidean":
            distance = nn.functional.pairwise_distance(e1, e2).item()
        elif metric == "cosine":
            distance = 1 - nn.functional.cosine_similarity(e1, e2).item()
        else:
            raise ValueError(f"Unknown metric {metric}")
        distances[str(comp)] = distance
    return distances

def normalize_distances(distances):
    """
    Convert distances to similarity in [0, 1]
    """
    distances_list = torch.tensor(list(distances.values()), dtype=torch.float32)
    max_dist = distances_list.max()
    similarities = 1 - distances_list / max_dist
    return similarities.tolist()

def compute_embeddings(model, data, batch_size=1000):
    embeddings_list = []
    cur_batch = []
    with torch.no_grad(), torch.inference_mode():
        for cur_sess in tqdm(data.keys()):
            cur_batch.append(cur_sess)
            if len(cur_batch) >= batch_size:
                new_embeddings = get_embeddings(model, cur_batch)
                embeddings_list += list(zip(cur_batch, new_embeddings))
                cur_batch = []
        if len(cur_batch) > 0:
            new_embeddings = get_embeddings(model, cur_batch)
            embeddings_list += list(zip(cur_batch, new_embeddings))
    return dict(embeddings_list)

def filter_data(data) -> dict:
    # Drop sessions with <5 samples and users with <2 sessions
    to_delete = []
    for user_key, sessions in data.items():
        for sess_key, sess in sessions.items():
            if len(sess) < 5:
                to_delete.append((user_key, sess_key))
    for user_key, sess_key in to_delete:
        del data[user_key][sess_key]

    to_delete_users = []
    for user_key, sessions in data.items():
        if len(sessions) < 2:
            to_delete_users.append(user_key)
    for user_key in to_delete_users:
        del data[user_key]
    return data

if __name__ == "__main__":
    pl.seed_everything(conf.seed, workers=True)

    dev_file_loc = '../../KVC_data/{}/{}_dev_set.npy'.format(conf.scenario, conf.scenario)
    data_file_loc = '../../KVC_data/{}/{}_test_sessions.npy'.format(conf.scenario, conf.scenario)
    comps_file_loc = "../../KVC_data/{}/{}_comparisons.txt".format(conf.scenario, conf.scenario)

    with open(comps_file_loc, "r") as file:
        comps = eval(file.readline())

    data = np.load(data_file_loc, allow_pickle=True).item()
    dev_data = np.load(dev_file_loc, allow_pickle=True).item()

    dev_data = filter_data(dev_data)
    users = list(dev_data.keys())
    random.Random(conf.seed).shuffle(users)

    n_total = len(users)
    n_train = int(n_total * 0.8)
    train_users = users[:n_train]
    train_data = {u: dev_data[u] for u in train_users}

    min_max_quantile = compute_feature_quantiles(dev_data)
    init_periods = compute_init_periods(min_max_quantile, 16)
    checkpoint_path = '../scripts/keystroke-ca/20251111_0149/checkpoints/classicLearnPeriodsKeyEmb_mobile-1931-1.45.ckpt'
    model = CNN_LTE(init_periods)
    loaded_model = load_model(model, checkpoint_path)
    embeddings = compute_embeddings(loaded_model, data)
    distances = compute_distances(embeddings, comps, metric="euclidean")

    similarities = normalize_distances(distances)

    with open('{}_predictions.txt'.format(conf.scenario), "w") as file:
        file.write(str(similarities))