import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    def __init__(self, item_num, params):
        super(BPRMF, self).__init__()
        self.n_factors = params['item_embedding_dim']
        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu' # cuda:0
        self.lr = params['learning_rate'] # 1e-4
        self.wd = params['weight_decay'] # 5e-4


        self.n_items = item_num + 1 # 多一个0代表空
        self.item_embedding = nn.Embedding(self.n_items, self.n_factors, padding_idx=0) # default embedding for item 0 is all zeros
        self.bn = nn.BatchNorm1d(self.n_factors)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
                
    def forward(self, seq, lengths):
        item_seq_emb = self.item_embedding(seq)

        uF = torch.div(
            torch.sum(item_seq_emb, dim=1), # (B,max_len,dim) -> (B,dim)
            lengths.float().unsqueeze(dim=1) # B -> B,1
        ) # (B, dim)
        item_embs = self.item_embedding(torch.arange(self.n_items)) # predict for all items, (n_item, dim)
        uF = self.bn(uF)
        scores = torch.matmul(uF, item_embs.transpose(0, 1)) # (B, n_item)

        return scores


    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)
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

                logit_sampled = logit[:, target.view(-1)] # 选出这一组batch中各个batch的next item groundtruth，对于这个方阵，对角线的位置是对应的真正GT

                # differences between the item scores
                diff = logit_sampled.diag().view(-1, 1).expand_as(logit_sampled) - logit_sampled # positive - negative
                loss = -torch.mean(F.logsigmoid(diff)) # BPR loss

                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                sample_num += target.numel()
            
            # self.scheduler.step()

            if valid_loader is not None:
                self.eval()
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

            train_time = time.time() - start_time
            print(f'training epoch [{epoch}/{self.epochs}]\tTrain Loss: {total_loss:.4f} \tTrain Elapse: {train_time:.2f}s')
            if valid_loader is not None:
                print(f'Test Metrics: HR@10: {res_hr:.4f}, MRR@10: {res_mrr:.4f}, NDCG@10: {res_ndcg:.4f}')
            
    def predict(self, test_loader, k:list=[10]):
        self.eval()

        preds = {topk : torch.tensor([]) for topk in k}
        last_item = torch.tensor([])

        for seq, target, lens in test_loader:
            seq = seq.to(self.device)
            lens = lens.to(self.device)

            scores = self.forward(seq, lens) # B, n_item, here n_item=true item num + 1
            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 # [:, 1:] to delect item 0, +1 to represent the actual code of items

            for topk in k:
                preds[topk] = torch.cat((preds[topk], rank_list[:, :topk].cpu()), 0)
        
            last_item = torch.cat((last_item, target), 0)

        return preds, last_item