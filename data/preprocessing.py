import numpy as np
import random
import torch


def extract_features_classic(session):
    """
    Extract keystroke dynamics features from a session.
    session: np.ndarray of shape (N, 3) with columns [press_time, release_time, key_code]
    """

    try:
        press = session[:, 0].astype(np.float64) / 1000.0
        release = session[:, 1].astype(np.float64) / 1000.0
        ascii_f = session[:, 2].astype(int)

        hold = release - press
        flight_time = np.zeros_like(press)
        flight_time[1:] = press[1:] - release[:-1]

        numeric_feats = np.column_stack([hold, flight_time, ascii_f])
    except:
        print(session)
        print(session.shape)

    return numeric_feats

def get_session_fixed_length(session, sequence_length, start_zero=True):
    num_tile = max(1, sequence_length // session.shape[0]) + 2
    tiled = np.tile(session, (num_tile, 1))
    if start_zero:
        start_idx = 0
    else:
        start_idx = np.random.randint(0, session.shape[0] - 1)
    # print(tiled.shape, start_idx, sequence_length, tiled[start_idx:start_idx+sequence_length].shape)
    return tiled[start_idx:start_idx+sequence_length]

def get_session_fixed_length_zero_pad_with_mask(session, sequence_length, start_zero=True):
    T, D = session.shape
    mask = np.zeros(sequence_length, dtype=np.bool_)

    if T >= sequence_length:
        start_idx = 0 if start_zero else np.random.randint(0, T - sequence_length + 1)
        mask[:] = True
        return session[start_idx:start_idx + sequence_length], mask

    padded = np.zeros((sequence_length, D), dtype=session.dtype)
    padded[:T] = session
    mask[:T] = 1 #True
    return padded, mask


def get_session_fixed_length_zero_pad(session, sequence_length, start_zero=True):
    T, D = session.shape

    # Case 1: session is long enough → crop
    if T >= sequence_length:
        if start_zero:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, T - sequence_length + 1)
        return session[start_idx:start_idx + sequence_length]

    # Case 2: session is too short → zero-pad
    padded = np.zeros((sequence_length, D), dtype=session.dtype)
    padded[:T] = session
    return padded

def augment_session(session):
    new_session = []
    for cur_event in session.copy():
        if random.random() < 0.02:
            # drop the event
            continue
        if random.random() < 0.1:
            # copy the event
            new_session.append(cur_event.copy())
        if random.random() < 0.1:
            # change the start time
            cur_event[0] += (random.random() - 0.5) * 100
        if random.random() < 0.1:
            # change the end time
            cur_event[1] += (random.random() - 0.5) * 100
        if random.random() < 0.1:
            # change the symbol
            cur_event[2] = random.randint(10, 255)
        # if random.random() < 0.1:
        #     # zero the event
        #     cur_event = np.zeros_like(cur_event)
        new_session.append(cur_event)
    if len(new_session) < 3:
        return session
    return np.array(new_session)


# def augment_session(session, seed=None):
#     new_session = []
#
#     for cur_event in session.copy():
#         if random.random() < 0.2:  # higher chance for realistic noise
#             hold = cur_event[1] - cur_event[0]
#             jitter = random.gauss(0, hold * 0.05)  # 5% std deviation
#             cur_event[0] += jitter
#             cur_event[1] += jitter
#         if random.random() < 0.05:
#             # drop the event
#             continue
#         if random.random() < 0.1:
#             # copy the event
#             new_session.append(cur_event.copy())
#         if random.random() < 0.1:
#             # change the start time
#             cur_event[0] += (random.random() - 0.5) * 100
#         if random.random() < 0.1:
#             # change the end time
#             cur_event[1] += (random.random() - 0.5) * 100
#         if random.random() < 0.1:
#             # change the symbol
#             cur_event[2] = random.randint(10, 255)
#         # if random.random() < 0.1:
#         #     # zero the event
#         #     cur_event = np.zeros_like(cur_event)
#         new_session.append(cur_event)
#     if len(new_session) < 3:
#         return session
#     return np.array(new_session)

def compute_init_periods(min_max_quantile: dict, n_periods: int):
    init_periods = {}
    for feat, (min_period, max_period) in min_max_quantile.items():
        min_period = max(min_period, 1e-4)
        init_periods[feat] = torch.tensor(
            np.logspace(np.log(min_period), np.log(max_period), n_periods),
            dtype=torch.float32
        )
    return init_periods

def compute_feature_quantiles(dataset):
    """
    dataset: dictionary {user_id: {session_id: np.ndarray}}
    extract_features: function to extract numeric features from a session
    Returns: dict with (min, max) per feature
    """
    all_features = []
    for user_sessions in dataset.values():
        for session in user_sessions.values():
            feats = extract_features_classic(session)
            all_features.append(feats)

    all_features = np.vstack(all_features)  # shape: (total_keystrokes, num_features)
    feature_names = ['hold', 'flight']

    # feature_names = ['hold', 'press_to_press', 'flight', 'release_to_release', 'rel_press', 'rel_release']
    clip_dict = {}
    for i, name in enumerate(feature_names):
        quantile_values = np.quantile(all_features[:, i], [0.01, 0.99])
        clip_dict[name] = np.round(quantile_values, 4)
        print(f"{name}: 1% = {quantile_values[0]:.4f}, 99% = {quantile_values[1]:.4f}")
    return clip_dict

