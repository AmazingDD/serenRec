import time
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

class NARM(nn.Module):
    def __init__(self, item_num, params):
        super(NARM, self).__init__()

        self.logger = params['logger']

        self.embedding_size = params['item_embedding_dim']
        self.hidden_size = self.embedding_size # params['hidden_size']
        self.n_layers = params['num_layers']
        self.dropout_prob = params['dropout_prob']

        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_prob)
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]) # (B)->(B, 1, D)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)

        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        seq_output = self.b(c_t)

        return seq_output
    
    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)

        self.best_state_dict = None
        self.best_kpi = -1
        for epoch in range(1, self.epochs + 1):
            self.train()

            total_loss = 0.
            sample_num = 0

            start_time = time.time()
            for seq, target, lens in tqdm(train_loader, desc='Training', unit='batch'):
                self.optimizer.zero_grad()
                seq = seq.to(self.device) # (B,max_len)
                target = target.to(self.device) # (B)
                lens = lens.to(self.device) # (B)

                seq_output = self.forward(seq, lens) # (B, D)
                test_item_emb = self.item_embedding.weight # (N, D)
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # (B, N)
                loss = F.cross_entropy(logits, target)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                sample_num += target.numel()

            train_time = time.time() - start_time
            self.logger.info(f'Training epoch [{epoch}/{self.epochs}]\tTrain Loss: {total_loss:.4f}\tTrain Elapse: {train_time:.2f}s')

            if valid_loader is not None:
                start_time = time.time()
                with torch.no_grad():
                    res_kpis = self.evaluate(valid_loader, [10])
                    res_mrr = res_kpis[10]['MRR']
                    res_hr = res_kpis[10]['HR']
                    res_ndcg = res_kpis[10]['NDCG']

                if self.best_kpi < res_mrr:
                    self.best_state_dict = deepcopy(self.state_dict())
                    self.best_kpi = res_mrr
                valid_time = time.time() - start_time
                self.logger.info(f'Valid Metrics: HR@10: {res_hr:.4f}\tMRR@10: {res_mrr:.4f}\tNDCG@10: {res_ndcg:.4f}\tValid Elapse: {valid_time:.2f}s')

    def predict(self, test_loader, k:list=[10]):
        self.eval()

        preds = {topk : torch.tensor([]) for topk in k}
        last_item = torch.tensor([])

        for seq, target, lens in tqdm(test_loader, desc='Testing', unit='batch'): 
            seq = seq.to(self.device)
            lens = lens.to(self.device)

            seq_output = self.forward(seq, lens) 
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # (B, D), (N, D)->(B, N)

            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 

            for topk in k:
                preds[topk] = torch.cat((preds[topk], rank_list[:, :topk].cpu()), 0)
        
            last_item = torch.cat((last_item, target), 0)

        return preds, last_item
    
    def evaluate(self, test_loader, k:list=[10]):
        self.eval()

        res = {topk : {'MRR': 0., 'NDCG': 0., 'HR': 0.,} for topk in k}
        batch_cnt = 0

        for seq, target, lens in tqdm(test_loader, desc='Testing', unit='batch'): 
            seq = seq.to(self.device)
            lens = lens.to(self.device)
            target = target.to(self.device)

            seq_output = self.forward(seq, lens) 
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # (B, D), (N, D)->(B, N)
            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 

            batch_cnt += 1
            for topk in k:
                pred = rank_list[:, :topk]
                B, topk = pred.size()
                expand_target = target.unsqueeze(1).expand(-1, topk)
                hr = (pred == expand_target)
                ranks = (hr.nonzero(as_tuple=False)[:, -1] + 1).float()
                mrr = torch.reciprocal(ranks)
                ndcg = 1 / torch.log2(ranks + 1)

                res_hr = hr.sum(axis=1).float().mean().item()
                res_mrr = torch.cat([mrr, torch.zeros(B - len(mrr), device=self.device)]).mean().item()
                res_ndcg = torch.cat([ndcg, torch.zeros(B - len(ndcg), device=self.device)]).mean().item()

                res[topk]['MRR'] += res_mrr
                res[topk]['NDCG'] += res_ndcg
                res[topk]['HR'] += res_hr

        for topk in k:
            res[topk] = {kpi: r / batch_cnt for kpi, r in res[topk].items()}
        return res
