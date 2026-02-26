import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.Aalto.preprocessing import (augment_session, compute_feature_min_max,
                                      extract_features_classic, get_session_fixed_length_zero_pad_with_mask)
import numpy as np
from sklearn.preprocessing import StandardScaler

from typing import Dict, List, Tuple, Any
import torch

SessionData = np.ndarray
UserSessionDict = Dict[Any, Dict[Any, SessionData]]
UserPairs = List[Tuple[Any, Any, Any, Any, Tuple]]

class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        raw_data: np.array,
        predict_file_path: str = None,
        window_size: int,
        samples_per_batch_train: int,
        samples_per_batch_val: int,
        batches_per_epoch_train: int,
        batches_per_epoch_val: int,
        num_workers_train: int = 8,
        num_workers_val: int = 4,
        persistent_workers: bool = True,
        train_val_division: float = 0.8,
        augment: bool = False,
        seed: int = 42,
        min_session_length: int = 5,
        min_sessions_per_user: int = 2
    ):
        super().__init__()
        self.raw_data = raw_data
        self.predict_file_path = predict_file_path
        self.sequence_length = window_size
        self.samples_per_batch_train = samples_per_batch_train
        self.samples_per_batch_val = samples_per_batch_val
        self.batches_per_epoch_train = batches_per_epoch_train
        self.batches_per_epoch_val = batches_per_epoch_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.persistent_workers = persistent_workers
        self.train_val_division = train_val_division
        self.augment = augment
        self.seed = seed

        self._train_users = None
        self._val_users = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.ds_predict = None
        self.min_session_length = min_session_length
        self.min_sessions_per_user = min_sessions_per_user

    @staticmethod
    def _filter_data(
            data: UserSessionDict,
            min_session_length: int,
            min_sessions_per_user: int
    ) -> UserSessionDict:
        """
        Filter out:
        1. Sessions shorter than min_session_length
        2. Users with fewer than min_sessions_per_user sessions

        Args:
            data: Nested dict {user_id: {session_id: session_data}}
            min_session_length: Minimum keystrokes per session
            min_sessions_per_user: Minimum sessions per user

        Returns:
            Filtered data dictionary
        """
        # Deep copy to avoid modifying original
        filtered = {u: dict(sessions) for u, sessions in data.items()}

        # Remove short sessions
        for user_id in list(filtered.keys()):
            sessions = filtered[user_id]
            filtered[user_id] = {
                sid: sess for sid, sess in sessions.items()
                if len(sess) >= min_session_length
            }

        # Remove users with too few sessions
        filtered = {
            user_id: sessions for user_id, sessions in filtered.items()
            if len(sessions) >= min_sessions_per_user
        }

        return filtered

    def setup(self, stage: str|None = None) -> None:
        if stage in ("fit", None):
            # Split by users
            self.data = self._filter_data(self.raw_data, self.min_session_length, self.min_sessions_per_user)
            users = list(self.data.keys())
            random.Random(self.seed).shuffle(users)

            n_total = len(users)
            n_train = int(n_total * self.train_val_division)
            # n_val = int(n_total * 0.1)

            self._train_users = users[:n_train]
            self._val_users = users[n_train:]
            # self._test_users = users[n_train + n_val:]

            print('Train num users', len(self._train_users))
            print('Validation num users', len(self._val_users))
            # print('Test num users', len(self._test_users))

            self.train_data = {u: self.data[u] for u in self._train_users}
            self.val_data = {u: self.data[u] for u in self._val_users}
            # self.test_data = {u: data[u] for u in self._test_users}
            self.scaler = self._fit_scaler(self.train_data)

            # self.val_shared_pairs = build_cross_user_sequence_pairs(self.val_data)
            self.min_max = compute_feature_min_max(self.train_data)

            self.ds_train = PrepareData(
                self.train_data,
                window_size=self.sequence_length,
                samples_considered_per_epoch=self.batches_per_epoch_train * self.samples_per_batch_train,
                augment=self.augment,
                scaler=self.scaler
            )

            self.ds_val = PrepareData(
                self.val_data,
                window_size=self.sequence_length,
                samples_considered_per_epoch=self.batches_per_epoch_val * self.samples_per_batch_val,
                augment=False,
                scaler=self.scaler
            )
            # self.ds_val_same_seq = SameSequenceContrastiveData(
            #     self.val_data,
            #     self.val_shared_pairs,
            #     sequence_length=self.sequence_length,
            #     augment=False
            # )

        if stage == "predict":
            # self.pred_data = np.load(self.predict_file_path, allow_pickle=True).item()
            # self.ds_predict = PreparePredictData(
            #     self.pred_data,
            #     window_size=self.window_size,
            # )
            self.ds_predict = self.ds_val

    @property
    def num_train_users(self) -> int:
        return len(self._train_users)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.samples_per_batch_train,
            num_workers=self.num_workers_train,
            persistent_workers=self.persistent_workers and self.num_workers_train > 0,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.samples_per_batch_val,
            num_workers=self.num_workers_val,
            persistent_workers=self.persistent_workers and self.num_workers_val > 0,
            shuffle=False
        )

    def val_same_sequence_dataloader(self):
        return DataLoader(
            self.ds_val_same_seq,
            batch_size=self.samples_per_batch_val,
            shuffle=False,
            num_workers=self.num_workers_val,
            persistent_workers=self.persistent_workers and self.num_workers_val > 0,
        )

    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.ds_test,
    #         batch_size=self.samples_per_batch_val,
    #         num_workers=self.num_workers_val,
    #         persistent_workers=self.persistent_workers and self.num_workers_val > 0,
    #     )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            batch_size=self.samples_per_batch_val,
            num_workers=self.num_workers_val,
            persistent_workers=self.persistent_workers and self.num_workers_val > 0,
            shuffle=False
        )

    def _fit_scaler(self, train_data):
        all_features = []

        for user_id, sessions in train_data.items():
            for session_id, session in sessions.items():
                features = extract_features_classic(session)  # (T, F)
                all_features.append(features)

        all_features = np.concatenate(all_features, axis=0)  # (total_T, F)

        scaler = StandardScaler()
        scaler.fit(all_features)

        return scaler


class PrepareData:
    def __init__(self, dataset, window_size, samples_considered_per_epoch, augment, scaler):
        self.data = dataset
        self.len = samples_considered_per_epoch
        self.window_size = window_size
        self.user_keys = list(self.data.keys())
        self.augment = augment
        self.scaler = scaler

    def __getitem__(self, index):
        # first user & session
        user_idx_1, user_key_1 = self._pick_random_user(self.user_keys)
        session_idx_1 = self._pick_random_session(user_key_1, exclude_idx=None)
        session_1, mask_1 = self._load_and_process_session(user_key_1, session_idx_1)

        # decide same/different
        same_user = random.random() < 0.5
        label = 0 if same_user else 1

        if same_user: # SAME USER
            user_idx_2, user_key_2 = user_idx_1, user_key_1
            session_idx_2 = self._pick_random_session(user_key_2, exclude_idx=session_idx_1)
        else: # DIFFERENT USERS
            user_idx_2, user_key_2  = self._pick_random_different_user(self.user_keys, user_idx_1)
            session_idx_2 = self._pick_random_session(user_key_2, exclude_idx=None)

        session_2, mask_2 = self._load_and_process_session(user_key_2, session_idx_2)
        return (
            (session_1, mask_1),
            (session_2, mask_2),
            label,
            (user_key_1, user_key_2),
            (torch.tensor(user_idx_1), torch.tensor(user_idx_2)) #(user_key_1, user_key_2),
        )

    def __len__(self):
        return self.len

    def _load_and_process_session(self, user_id, session_id):
        """Load raw session → optionally augment → fix length → feature extraction."""
        session = self.data[user_id][session_id]
        if self.augment:
            session = augment_session(session)

        prep_session = extract_features_classic(session)  # (T, F)

        # apply standardization
        prep_session = self.scaler.transform(prep_session)
        fix_session, mask = get_session_fixed_length_zero_pad_with_mask(prep_session, self.window_size, self.augment)

        return fix_session, mask

    @staticmethod
    def _pick_random_user(user_keys):
        """Pick a random user (optionally excluding one)."""
        possible_indices = list(range(len(user_keys)))
        idx = random.randrange(len(possible_indices))
        return idx, user_keys[idx]

    @staticmethod
    def _pick_random_different_user(user_keys, user_idx_1):
        """Return a random user != user_idx_1 using a fast while-loop."""
        user_idx_2 = user_idx_1
        while user_idx_2 == user_idx_1:
            user_idx_2 = random.randrange(len(user_keys))
        return user_idx_2, user_keys[user_idx_2]

    def _pick_random_session(self, user_id, exclude_idx=None):
        sessions = list(self.data[user_id].keys())
        if exclude_idx is not None:
            sessions = [s for s in sessions if s != exclude_idx]
        return random.choice(sessions)

class PreparePredictData:
    def __init__(self, dataset, sequence_length):
        self.sequence_length = sequence_length
        self.data = dataset
        self.samples = []

        for session_id in dataset.keys():
            self.samples.append(session_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        session_id = self.samples[index]
        session = self.data[session_id]

        prep_session = extract_features_classic(session)
        fix_session, mask = get_session_fixed_length_zero_pad_with_mask(
            prep_session,
            self.sequence_length,
            start_zero=True
        )
        return fix_session, mask, session_id

from collections import defaultdict
from itertools import combinations

class SameSequenceContrastiveData:
    def __init__(self, data, shared_pairs, sequence_length, augment=False):
        self.data = data
        self.shared_pairs = shared_pairs
        self.sequence_length = sequence_length
        self.augment = augment

        self.user_keys = list(self.data.keys())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_keys)}

    def __len__(self):
        return len(self.shared_pairs)

    def __getitem__(self, idx):
        u1, s1, u2, s2, _ = self.shared_pairs[idx]

        session_1, mask_1 = self._load(u1, s1)
        session_2, mask_2 = self._load(u2, s2)

        label = 1  # SAME SEQUENCE, DIFFERENT USER

        # In SameSequenceContrastiveData.__getitem__:
        return (
            (session_1, mask_1),
            (session_2, mask_2),
            label,
            (u1, u2),
            (torch.tensor(self.user_to_idx[u1]), torch.tensor(self.user_to_idx[u2])),  # Numeric indices for loss
        )

    def _load(self, user_id, session_id):
        session = self.data[user_id][session_id]

        if self.augment:
            session = augment_session(session)

        prep = extract_features_classic(session)
        fixed, mask = get_session_fixed_length_zero_pad_with_mask(
            prep, self.sequence_length, self.augment
        )

        return fixed, mask


def build_cross_user_sequence_pairs(data, key_col=2, min_unique_keys=5):
    """
    Returns a list of tuples:
    (user_id_1, session_id_1, user_id_2, session_id_2, sequence)
    """

    sequence_index = defaultdict(lambda: defaultdict(list))
    # seq -> user -> [session_ids]

    for user_id, sessions in data.items():
        for sess_id, sess in sessions.items():
            seq = tuple(int(x) for x in sess[:, key_col])

            # remove degenerate sessions
            if len(set(seq)) < min_unique_keys:
                continue

            sequence_index[seq][user_id].append(sess_id)

    pairs = []

    for seq, user_map in sequence_index.items():
        if len(user_map) < 2:
            continue

        for (u1, sids1), (u2, sids2) in combinations(user_map.items(), 2):
            for sid1 in sids1:
                for sid2 in sids2:
                    pairs.append((u1, sid1, u2, sid2, seq))

    return pairs