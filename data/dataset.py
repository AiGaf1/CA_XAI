import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import conf
from data.preprocessing import (get_session_fixed_length, augment_session, compute_feature_quantiles,
                                compute_init_periods, extract_features_classic, get_session_fixed_length_zero_pad_with_mask)
import numpy as np

class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        raw_data: np.array,
        predict_file_path: str = None,
        sequence_length: int,
        samples_per_batch_train: int,
        samples_per_batch_val: int,
        batches_per_epoch_train: int,
        batches_per_epoch_val: int,
        num_workers_train: int = 8,
        num_workers_val: int = 4,
        persistent_workers: bool = True,
        train_val_division: float = 0.8,
        augment: bool = True,
        seed: int = 42,
        min_session_length: int = 5
    ):
        super().__init__()
        self.raw_data = raw_data
        self.predict_file_path = predict_file_path
        self.sequence_length = sequence_length
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

    @staticmethod
    def _filter_data(data, min_session_length=5, min_len_sessions=2) -> dict:
        # Drop sessions with <5 samples and users with <2 sessions
        to_delete = []
        for user_key, sessions in data.items():
            for sess_key, sess in sessions.items():
                if len(sess) < min_session_length:
                    to_delete.append((user_key, sess_key))
        for user_key, sess_key in to_delete:
            del data[user_key][sess_key]

        to_delete_users = []
        for user_key, sessions in data.items():
            if len(sessions) < min_len_sessions:
                to_delete_users.append(user_key)
        for user_key in to_delete_users:
            del data[user_key]
        return data

    def setup(self, stage: str|None = None) -> None:
        if stage in ("fit", None):
            # Split by users
            self.data = self._filter_data(self.raw_data, self.min_session_length)
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

            min_max = compute_feature_quantiles(self.train_data)
            self.init_periods = compute_init_periods(min_max, conf.N_PERIODS)

            self.ds_train = PrepareData(
                self.train_data,
                sequence_length=self.sequence_length,
                samples_considered_per_epoch=self.batches_per_epoch_train * self.samples_per_batch_train,
                augment=self.augment
            )
            self.ds_val = PrepareData(
                self.val_data,
                sequence_length=self.sequence_length,
                samples_considered_per_epoch=self.batches_per_epoch_val * self.samples_per_batch_val,
                augment=False,
            )
        if stage == "predict":
            # self.pred_data = np.load(self.predict_file_path, allow_pickle=True).item()
            # self.ds_predict = PreparePredictData(
            #     self.pred_data,
            #     sequence_length=self.sequence_length,
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


class PrepareData:
    def __init__(self, dataset, sequence_length, samples_considered_per_epoch, augment):
        self.data = dataset
        self.len = samples_considered_per_epoch
        self.sequence_length = sequence_length
        self.user_keys = list(self.data.keys())
        self.augment = augment

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
            (user_idx_1, user_idx_2),
        )

    def __len__(self):
        return self.len

    def _load_and_process_session(self, user_id, session_id):
        """Load raw session → optionally augment → fix length → feature extraction."""
        session = self.data[user_id][session_id]
        if self.augment:
            session = augment_session(session)

        prep_session = extract_features_classic(session)
        fix_session, mask = get_session_fixed_length_zero_pad_with_mask(prep_session, self.sequence_length, self.augment)

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

