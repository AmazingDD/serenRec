import time
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import surrogate, neuron, functional, layer

class Scaser(nn.Module):
    def __init__(self, item_num, params):
        super(Scaser, self).__init__()
        self.logger = params['logger']
        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.max_seq_length = params['max_seq_len']
        self.T = params['T']

        self.embedding_size = params['item_embedding_dim']
        self.n_h = 8 # params["nh"] (int) The number of horizontal Convolutional filters.                   
        self.n_v = 4 # params["nv"] (int) The number of vertical Convolutional filters.
        self.reg_weight = params["reg_weight"]

        # for fair comparison, remove user embedding for session-recommendation
        # self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # vertical conv layer
        self.conv_v = layer.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1), bias=False)

        self.rpe_ln = nn.LayerNorm(self.embedding_size)
        self.rpe_lif = neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan())
        
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [layer.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size), bias=False) for i in lengths])
        self.bn_conv = nn.ModuleList([layer.BatchNorm2d(self.n_h) for _ in lengths])
        self.act_conv = nn.ModuleList([neuron.LIFNode(tau=params['tau'], detach_reset=True, surrogate_function=surrogate.ATan()) for _ in lengths])
        
        # fully-connected layer
        self.fc_dim_v = self.n_v * self.embedding_size
        self.fc_dim_h = self.n_h * len(lengths)
        self.fc_v = layer.Linear(self.fc_dim_v, self.embedding_size)
        self.fc_h = layer.Linear(self.fc_dim_h, self.embedding_size)

        functional.set_step_mode(self, 'm')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self, item_seq):
        B, _ = item_seq.shape
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.rpe_ln(item_seq_emb).unsqueeze(1)  # ->(B, 1, L, D)
        item_seq_emb = item_seq_emb.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # ->(T, B, 1, L, D)
        item_seq_emb = self.rpe_lif(item_seq_emb)  # ->(T, B, 1, L, D)

        # vertical conv Layers
        out_h, out_v = None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb) # ->(T, B, 1, L, D)->(T, B, nv, D)
            out_v = out_v.reshape(self.T, B, self.fc_dim_v) # -> (T, B, nv*D) 

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for i in range(self.max_seq_length):
                conv = self.conv_h[i]
                bn = self.bn_conv[i]
                lif = self.act_conv[i]
                conv_out = bn(conv(item_seq_emb)).squeeze(4) # -> (T, B, nh, L, 1) ->(T, B, nh, L) 
                conv_out = lif(conv_out) # ->(T, B, nh, L) 
                conv_out = conv_out.flatten(0, 1) # ->(TB, nh, L) 
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) # -> (TB, nh) 
                pool_out = pool_out.reshape(self.T, B, -1) # -> (T, B, nh) 
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 2)   # -> (T, B, seq_len*nh)  prepare for fully connect

        # fully-connected layer
        out_h = self.fc_h(out_h) # -> (T, B, D)
        out_h = out_h.mean(0) # -> (B, D)
        out_v = self.fc_v(out_v.mean(0)) 
        seq_output = out_h + out_v # ->(B, D)
        
        return seq_output # -> (B, D)
    
    def reg_loss_conv_h(self):
        """  L2 loss on conv_h """
        loss_conv_h = 0
        for name, parm in self.conv_h.named_parameters():
            if name.endswith("weight"):
                loss_conv_h = loss_conv_h + parm.norm(2)
        return self.reg_weight * loss_conv_h
    
    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)

        self.best_state_dict = None
        self.best_kpi = -1
        for epoch in range(1, self.epochs + 1):
            self.train()

            total_loss = 0.
            sample_num = 0

            start_time = time.time()
            for seq, target, _ in tqdm(train_loader, desc='Training', unit='batch'):
                self.optimizer.zero_grad()
                seq = seq.to(self.device) # (B,max_len)
                target = target.to(self.device) # (B)

                seq_output = self.forward(seq) # (B, D)
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = F.cross_entropy(logits, target)

                reg_loss = self.item_embedding.weight.norm(2) + self.conv_v.weight.norm(2) + self.fc_v.weight.norm(2) + self.fc_h.weight.norm(2)
                loss = loss + self.reg_loss_conv_h() + self.reg_weight * reg_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                sample_num += target.numel()

                functional.reset_net(self)

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

        for seq, target, _ in tqdm(test_loader, desc='Testing', unit='batch'):
            seq = seq.to(self.device)

            seq_output = self.forward(seq) 
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # (B, D), (N, D)->(B, N)

            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 

            for topk in k:
                preds[topk] = torch.cat((preds[topk], rank_list[:, :topk].cpu()), 0)
        
            last_item = torch.cat((last_item, target), 0)

            functional.reset_net(self)

        return preds, last_item

    def evaluate(self, test_loader, k:list=[10]):
        self.eval()

        res = {topk : {'MRR': 0., 'NDCG': 0., 'HR': 0.,} for topk in k}
        batch_cnt = 0

        for seq, target, _ in tqdm(test_loader, desc='Testing', unit='batch'): 
            seq = seq.to(self.device)
            target = target.to(self.device)

            seq_output = self.forward(seq) 
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
            functional.reset_net(self)

        for topk in k:
            res[topk] = {kpi: r / batch_cnt for kpi, r in res[topk].items()}
        return res