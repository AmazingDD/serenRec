import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class Interactions(object):
    def __init__(self, config, encoding=True) -> None:
        self.data = None
        self.user_num, self.item_num = None, None

        self.dataset = config['dataset']
        self.max_len = config['max_seq_len']
        self.uid_name = config['uid_name']
        self.iid_name = config['iid_name']
        self.inter_name = config['inter_name']
        self.time_name = config['time_name']
        self.test_ratio = config['test_ratio'] # 0.2
        self.prepro = config['prepro']

        self.encoding = encoding

    def _load_raw_data(self):
        data_dir = None
        if self.dataset == 'ml-25m':
            data_dir = f'./movielens/{self.dataset}/'
            self.data_dir = os.path.join(data_dir, 'ratings.csv')
            assert os.path.exists(self.data_dir), f'ratings.csv in {data_dir} not exists'
            self.data = pd.read_csv(self.data_dir)
        elif self.dataset == 'ml-1m':
            data_dir = f'./movielens/{self.dataset}/'
            self.data_dir = os.path.join(data_dir, 'ratings.dat')
            assert os.path.exists(self.data_dir), f'ratings.csv in {data_dir} not exists'
            self.data = pd.read_csv(self.data_dir, delimiter='::', names=[self.uid_name, self.iid_name, self.inter_name, self.time_name], engine='python')
        else:
            raise NameError(f'Invalid dataset name: {self.dataset}')
        
        self.data.drop_duplicates([self.uid_name, self.iid_name], keep='last', ignore_index=True)
        self.user_num = self.data[self.uid_name].nunique()
        self.item_num = self.data[self.iid_name].nunique()
        
    def _build_seq(self):
        if self.time_name is not None:
            self.data.sort_values(by=[self.uid_name, self.time_name], ascending=True, inplace=True)
        else:
            self.data.sort_values(ny=[self.uid_name], ascending=True, inplace=True)

        self.data.reset_index(drop=True, inplace=True)

        uid_col_arr = self.data[self.uid_name].to_numpy()
        iid_col_arr = self.data[self.iid_name].to_numpy()

        last_uid = None
        uid_list, items_list, target_list, item_list_length = [], [], [], []
        seq_start = 0

        for i, uid in enumerate(uid_col_arr):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > self.max_len:
                    seq_start += 1
                uid_list.append(uid)
                items_list.append(iid_col_arr[seq_start: i])
                target_list.append(iid_col_arr[i])
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_length = np.array(item_list_length, dtype=np.int64)
        
        new_length = len(items_list) # number of sequences
        self.item_list_len = torch.tensor(item_list_length)
        new_item_list = torch.zeros((new_length, self.max_len)).long()

        for i, (items, length) in enumerate(zip(items_list, item_list_length)):
            new_item_list[i][:length] = torch.LongTensor(items)

        self.target_list = torch.LongTensor(target_list)
        self.new_item_list = new_item_list

    def _filter_core(self): 
        # TODO
        # nead repeat
        user_item_counts = self.data.groupby('user')['item'].transform('count')
        item_user_counts = self.data.groupby('item')['user'].transform('count')

        filtered_df = self.data[
            (user_item_counts >= 5) & (item_user_counts >= 5)
        ]

    def _encode_id(self):
        token_uid = pd.Categorical(self.data[self.uid_name]).categories.to_numpy()
        token_iid = pd.Categorical(self.data[self.iid_name]).categories.to_numpy()
        # start from 1, 0 for none-item
        self.token_uid = {token + 1: uid for token, uid in enumerate(token_uid)}
        self.token_iid = {token + 1: iid for token, iid in enumerate(token_iid)}

        self.uid_token = {v: k for k, v in self.token_uid.items()}
        self.iid_token = {v: k for k, v in self.token_iid.items()}

        # tokenize columns
        self.data[self.uid_name] = self.data[self.uid_name].map(self.uid_token)
        self.data[self.iid_name] = self.data[self.iid_name].map(self.iid_token)

    def _build_dataset(self):
        rnd_idx = torch.randperm(len(self.target_list))
        split_point = int(len(self.target_list) * (1 - self.test_ratio))
        train_idx, test_idx = rnd_idx[:split_point], rnd_idx[split_point:]
        self.train_data = [self.new_item_list[train_idx], self.target_list[train_idx], self.item_list_len[train_idx]]
        self.test_data = [self.new_item_list[test_idx], self.target_list[test_idx], self.item_list_len[test_idx]]

    def build(self):
        self._load_raw_data()
        self._encode_id()

        if self.prepro:
            self._filter_core()

        self._build_seq()
        self._build_dataset()


def get_dataloader(ds, batch_size, shuffle, num_workers=4):  
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class SequentialDataset(Dataset):
    def __init__(self, data):
        super(SequentialDataset, self).__init__()
        self.seqs = data[0]
        self.next_item = data[1]
        self.seq_lens = data[3]

    def __len__(self):
        return len(self.next_item)

    def __getitem__(self, index):
        return self.seqs[index], self.next_item[index], self.seq_lens[index]
