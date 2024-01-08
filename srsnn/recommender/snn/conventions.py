import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate, functional


class SMF(nn.Module):
    ''' Spike MF with BPR loss '''
    def __init__(self, item_num, params):
        super(SMF, self).__init__()
        self.n_factors = params['item_embedding_dim']
        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu' # cuda:0
        self.lr = params['learning_rate'] # 1e-4
        self.wd = params['weight_decay'] # 5e-4
        self.T = params['T']

        self.n_items = item_num + 1 
        self.item_embedding = nn.Embedding(self.n_items, self.n_factors, padding_idx=0) 

        self.lif = neuron.LIFNode(tau=params['tau'], v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.ln = nn.LayerNorm(self.n_factors)

        functional.set_step_mode(self, 'm')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
                
    def forward(self, seq, lengths):
        item_seq_emb = self.item_embedding(seq) # (B, L, D)
        B, L, D = item_seq_emb.shape
        item_seq_emb = item_seq_emb.unsqueeze(0).repeat(self.T, 1, 1, 1) # (T, B, L, D)
        
        uF = torch.sum(self.ln(item_seq_emb.flatten(0, 1)).reshape(self.T, B, L, D), dim=2) # (T, B, D)
        uF = torch.div(uF, lengths.float().unsqueeze(1).repeat(self.T, 1, 1))  
        uF = self.lif(uF) # (T, B, D)
        uF = uF.mean(0) # (B, D)

        test_item_emb = self.item_embedding.weight # (n_item, D)
        scores = torch.matmul(uF, test_item_emb.transpose(0, 1)) # (B, n_item)

        return scores


    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)

        self.best_state_dict = None
        best_kpi = -1
        for epoch in range(1, self.epochs + 1):
            self.train()

            total_loss = 0.
            sample_num = 0

            start_time = time.time()
            for seq, target, lens in train_loader:
                self.optimizer.zero_grad()

                seq = seq.to(self.device) # (B,max_len)
                target = target.to(self.device) # (B)
                lens = lens.to(self.device) # (B)

                logit = self.forward(seq, lens)
                logit_sampled = logit[:, target.view(-1)]

                diff = logit_sampled.diag().view(-1, 1).expand_as(logit_sampled) - logit_sampled
                loss = -torch.mean(F.logsigmoid(diff)) # BPR loss

                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                sample_num += target.numel()

                functional.reset_net(self)

            self.scheduler.step()

            train_time = time.time() - start_time
            print(f'Training epoch [{epoch}/{self.epochs}]\tTrain Loss: {total_loss:.4f}\tTrain Elapse: {train_time:.2f}s')

            if valid_loader is not None:
                start_time = time.time()
                with torch.no_grad():
                    preds, last_item = self.predict(valid_loader, [10])
                    pred = preds[10]
                    N, topk = pred.size()
                    expand_target = last_item.unsqueeze(1).expand(-1, topk)
                    hr = (pred == expand_target)
                    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
                    mrr = torch.reciprocal(ranks)
                    ndcg = 1 / torch.log2(ranks + 1)

                    res_hr = hr.sum(axis=1).float().mean().item()
                    res_mrr = torch.cat([mrr, torch.zeros(N - len(mrr))]).mean().item()
                    res_ndcg = torch.cat([ndcg, torch.zeros(N - len(ndcg))]).mean().item()
                if best_kpi < res_mrr:
                    self.best_state_dict = self.state_dict()
                    best_kpi = res_mrr
                valid_time = time.time() - start_time
                print(f'Valid Metrics: HR@10: {res_hr:.4f}\tMRR@10: {res_mrr:.4f}\tNDCG@10: {res_ndcg:.4f}\tValid Elapse: {valid_time:.2f}s')
            
    def predict(self, test_loader, k:list=[15]):
        self.eval()

        preds = {topk : torch.tensor([]) for topk in k}
        last_item = torch.tensor([])

        for seq, target, lens in test_loader:
            seq = seq.to(self.device)
            lens = lens.to(self.device)

            scores = self.forward(seq, lens) # B, n_item, here n_item=true item num + 1
            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 # [:, 1:] to delect item 0, +1 to represent the actual code of items

            for topk in k:
                preds[topk] = torch.cat((preds[topk], rank_list[:,:topk].cpu()), 0)
        
            last_item = torch.cat((last_item, target), 0)

            functional.reset_net(self)

        return preds, last_item