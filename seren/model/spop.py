'''
@article{ludewig2018evaluation,
  title={Evaluation of session-based recommendation algorithms},
  author={Ludewig, Malte and Jannach, Dietmar},
  journal={User Modeling and User-Adapted Interaction},
  volume={28},
  number={4},
  pages={331--390},
  year={2018},
  publisher={Springer}
}
'''
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class SessionPop(nn.Module):
    def __init__(self, config):
        super(SessionPop, self).__init__()
        self.item_num = config['item_num']
        self.device = config['device']
        self.item_cnt_ref = torch.zeros(1 + config['item_num']).to(self.device) # the index starts from 1
        self.max_len = config['max_len']
        self.item_score = None

    def forward(self, item_seq):
        #predict as in the classic popularity model:
        idx, cnt = torch.unique(item_seq, return_counts=True)
        return idx, cnt

    def fit(self, train_loader):
        self.to(self.device)
        self.train()

        pbar = tqdm(train_loader)
        for btch in pbar:
            item_seq = btch[0]
            item_seq = item_seq.to(self.device)
            idx, cnt = self.forward(item_seq)
            self.item_cnt_ref[idx] += cnt
        self.item_score =  self.item_cnt_ref / (1+self.item_cnt_ref)

    def predict(self, input_ids, next_item):
        self.eval()

        input_ids = torch.tensor(input_ids).to(self.device)
        next_item = torch.tensor(next_item).to(self.device)
    
        item_cnt_ref = self.item_cnt_ref.clone()
        seq_max_len = torch.zeros(self.max_len).to(self.device)
        input_ids = pad_sequence([input_ids, seq_max_len])[0]
        
        idx, cnt = torch.unique(input_ids, return_counts=True)
        item_cnt_ref[idx] += (cnt + self.item_score[idx])
        return item_cnt_ref[next_item].detach().cpu().item()


    def rank(self, test_loader, topk=None, candidates=None):
        self.eval()

        res_ids, res_scs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for btch in pbar:
                item_seq = btch[0]
                item_seq = item_seq.to(self.device)
                scores = self.item_score.clone()
                scores = scores.tile((item_seq.shape[0], 1))
                # target = torch.zeros_like(item_score)
                cnt = torch.ones_like(scores)
                scores = scores.scatter_add_(1, item_seq, cnt)
                scs, ids = torch.sort(scores[:, 1:], descending=True)
                ids += 1
                
                if topk is not None and topk <= self.item_num:
                    ids, scs = ids[:, :topk], scs[:, :topk]
                res_ids = torch.cat((res_ids, ids), 0)
                res_scs = torch.cat((res_scs, scs), 0)

        return res_ids.detach().cpu(), res_scs.detach().cpu()
