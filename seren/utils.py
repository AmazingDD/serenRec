import re
import os
import gzip
import datetime
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def check_data_dir(root_path, data_filename):
    data_dir = root_path
    ensure_dir(data_dir)
    data_dir = os.path.join(data_dir, data_filename)
    assert os.path.exists(data_dir), f'{data_filename} in {root_path} not exists'
    return data_dir

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

class Logger:
    ''' spikingjelly induce logging not work '''
    def __init__(self, config, desc=None):
        log_root = config['log_path']
        dir_name = os.path.dirname(log_root)
        ensure_dir(dir_name)

        out_dir = os.path.join(log_root, f'{config["dataset"]}_{config["prepro"]}_{config["model"]}_T{config["T"]}_len{config["max_seq_len"]}_b{config["batch_size"]}_{config["epochs"]}epochs')

        ensure_dir(out_dir)

        logfilename = f'{desc}_record.log' # _{get_local_time()}
        logfilepath = os.path.join(out_dir, logfilename)

        self.filename = logfilepath

        f = open(logfilepath, 'w', encoding='utf-8')
        f.write(str(config) + '\n')
        f.flush()
        f.close()


    def info(self, s=None):
        print(s)
        f = open(self.filename, 'a', encoding='utf-8')
        f.write(f'[{get_local_time()}] - {s}\n')
        f.flush()
        f.close()

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
        self.sm = config['split_method']
        self.prepro = config['prepro']

        self.encoding = encoding

    def _load_raw_data(self):
        data_dir = None
        if self.dataset == 'ml-1m':
            # this dataset is only for toy-implementation
            self.data_dir = check_data_dir(f'./movielens/{self.dataset}/', 'ratings.dat')
            self.data = pd.read_csv(
                self.data_dir, 
                delimiter='::', 
                names=[self.uid_name, self.iid_name, self.inter_name, self.time_name], 
                engine='python')
            
        elif self.dataset == 'ml-25m':
            self.data_dir = check_data_dir(f'./movielens/{self.dataset}/', 'ratings.csv')
            self.data = pd.read_csv(self.data_dir)
            self.data.rename(
                columns={
                    'userId': self.uid_name, 
                    'movieId': self.iid_name, 
                    'rating': self.inter_name,
                    'timestamp': self.time_name}, 
                inplace=True)

        elif self.dataset == 'music':
            self.data_dir = check_data_dir(f'./amazon/', 'Digital_Music.csv')
            self.data = pd.read_csv(self.data_dir, 
                                    names=[self.iid_name, self.uid_name, self.inter_name, self.time_name])
        elif self.dataset == 'video':
            self.data_dir = check_data_dir(f'./amazon/', 'Video_Games.csv')
            self.data = pd.read_csv(self.data_dir, 
                                    names=[self.iid_name, self.uid_name, self.inter_name, self.time_name])
            
        elif self.dataset == 'arts':
            self.data_dir = check_data_dir(f'./amazon/', 'Arts_Crafts_and_Sewing.csv')
            self.data = pd.read_csv(self.data_dir, 
                                    names=[self.iid_name, self.uid_name, self.inter_name, self.time_name])
            
        elif self.dataset == 'steam':
            self.data_dir = check_data_dir(f'./steam/', 'steam_reviews.json.gz')

            i = 0
            df = {}
            for d in parse(self.data_dir):
                df[i] = d
                i += 1
            self.data = pd.DataFrame.from_dict(df, orient='index')
            self.data = self.data[['user_id', 'product_id', 'hours', 'date']].copy()
            self.data = self.data[~self.data['user_id'].isna()].reset_index(drop=True)
            self.data.rename(
                columns={
                    'user_id': self.uid_name, 
                    'product_id': self.iid_name, 
                    'hours': self.inter_name, 
                    'date': self.time_name}, 
                inplace=True)
            self.data[self.inter_name] = self.data[self.inter_name].fillna(0.)

        elif self.dataset == 'retail':
            self.data_dir = check_data_dir(f'./retail/', 'events.csv')

            self.data = pd.read_csv(self.data_dir, header=0, usecols=[0, 1, 2, 3], dtype={0:np.int64, 1:np.int64, 2:str, 3:np.int64})
            self.data.columns = [self.time_name, self.uid_name, 'event', self.iid_name]
            self.data[self.time_name] = (self.data[self.time_name] / 1000).astype(int)
            self.data[self.time_name] = pd.to_datetime(self.data[self.time_name], unit='s')
            self.data = self.data.query('event == "view"').reset_index(drop=True)
            del self.data['event']

        elif self.dataset == 'yoochoose':
            self.data_dir = check_data_dir(f'./yoochoose/', 'yoochoose-clicks.dat')
            self.data = pd.read_csv(
                self.data_dir,
                sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64},
                names=[self.uid_name, self.time_name, self.iid_name] #'SessionId', 'time', 'ItemId'
            )
            self.data[self.time_name] = pd.to_datetime(self.data[self.time_name], format='%Y-%m-%dT%H:%M:%S.%fZ')

        else:
            raise NameError(f'Invalid dataset name: {self.dataset}')
        
        self.data.drop_duplicates([self.uid_name, self.iid_name], keep='last', ignore_index=True)
        
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

        self.uid_list = np.array(uid_list)
        item_list_length = np.array(item_list_length, dtype=np.int64)
        
        new_length = len(items_list) # number of sequences
        self.item_list_len = torch.tensor(item_list_length)
        new_item_list = torch.zeros((new_length, self.max_len)).long()

        for i, (items, length) in enumerate(zip(items_list, item_list_length)):
            new_item_list[i][:length] = torch.LongTensor(items)

        self.target_list = torch.LongTensor(target_list)
        self.new_item_list = new_item_list

    def _filter_core(self): 
        if self.prepro == 'raw':
            pass
        elif self.prepro.endswith('core'):
            pattern = re.compile(r'\d+')
            core_num = int(pattern.findall(self.prepro)[0])
            # 5core default
            while True:
                user_item_counts = self.data.groupby('user')['item'].transform('count')
                item_user_counts = self.data.groupby('item')['user'].transform('count')

                filtered_df = self.data[
                    (user_item_counts >= core_num) & (item_user_counts >= core_num)
                ]

                if len(filtered_df) < len(self.data):
                    self.data = filtered_df
                else:
                    break
        else:
            raise ValueError('Invalid prepro value...')

        self.data.reset_index(drop=True, inplace=True)

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

        self.user_num = self.data[self.uid_name].nunique()
        self.item_num = self.data[self.iid_name].nunique()

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index.values()

    def _build_dataset(self, session=True):
        train_idx, test_idx = [], []
        if self.sm == 'ufo':
            print('fold-out by user')
            grouped_index = self._grouped_index(self.uid_list)
            for grouped_ids in grouped_index:
                total_cnt = len(grouped_ids)
                split_ids = int(total_cnt * (1 - self.test_ratio))
                train_idx.extend(grouped_ids[0:split_ids])
                test_idx.extend(grouped_ids[split_ids:total_cnt])
        elif self.sm == 'fo':
            print('random fold-out')
            rnd_idx = torch.randperm(len(self.target_list))
            split_point = int(len(self.target_list) * (1 - self.test_ratio))
            train_idx, test_idx = rnd_idx[:split_point], rnd_idx[split_point:]
        else:
            raise ValueError(f'Invalid train test split method: {self.sm}')

        if session:
            self.train_data = [self.new_item_list[train_idx], self.target_list[train_idx], self.item_list_len[train_idx]]
            self.test_data = [self.new_item_list[test_idx], self.target_list[test_idx], self.item_list_len[test_idx]]
        else: 
            self.train_data = [self.uid_list[train_idx], self.new_item_list[train_idx], self.target_list[train_idx], self.item_list_len[train_idx]]
            self.test_data = [self.uid_list[test_idx], self.new_item_list[test_idx], self.target_list[test_idx], self.item_list_len[test_idx]]

    def build(self):
        self._load_raw_data()
        print('Finish load raw data')
        self._filter_core()
        print(f'Finish {self.prepro} processing')
        self._encode_id()
        print(f'Finish re-encoding iid and uid')

        self._build_seq()
        print(f'Finish building sequences from original data')
        self._build_dataset()
        print('Finish load data')


def get_dataloader(ds, batch_size, shuffle, num_workers=4):  
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class SequentialDataset(Dataset):
    def __init__(self, data):
        super(SequentialDataset, self).__init__()
        self.seqs = data[0]
        self.next_item = data[1]
        self.seq_lens = data[2]

    def __len__(self):
        return len(self.next_item)

    def __getitem__(self, index):
        return self.seqs[index], self.next_item[index], self.seq_lens[index]
    

def accuracy_calculator(pred, last_item):
    N, topk = pred.size()
    expand_target = last_item.unsqueeze(1).expand(-1, topk)

    hr = (pred == expand_target)
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float() # ranking from 1, but return index for nonzero start with 0
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1 / torch.log2(ranks + 1)

    metrics = {
        'HR': hr.sum(axis=1).float().mean().item(),
        'MRR': torch.cat([mrr, torch.zeros(N - len(mrr))]).mean().item(), # no last_item in pred means the mrr/ndcg is zero for them
        'NDCG': torch.cat([ndcg, torch.zeros(N - len(ndcg))]).mean().item()
    }

    return metrics