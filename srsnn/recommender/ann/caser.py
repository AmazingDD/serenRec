import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class Caser(nn.Module):
    def __init__(self, item_num, params):
        super(Caser, self).__init__()

        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.max_seq_length = params['max_seq_len']

        self.embedding_size = params['item_embedding_dim']
        self.n_h = 8 # params["nh"] (int) The number of horizontal Convolutional filters.                   
        self.n_v = 4 # params["nv"] (int) The number of vertical Convolutional filters.
        self.dropout_prob = params["dropout_prob"]
        self.reg_weight = params["reg_weight"]

        # for fair comparison, remove user embedding for session-recommendation
        # self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # vertical conv layer
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1)
        )

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size)) for i in lengths])
        
        # fully-connected layer
        self.fc_dim_v = self.n_v * self.embedding_size
        self.fc_dim_h = self.n_h * len(lengths)
        fc_dim_in = self.fc_dim_v + self.fc_dim_h
        self.fc = nn.Linear(fc_dim_in, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.act_conv = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1) # ->(B, 1, L, D)
        # user_emb = self.user_embedding(user).squeeze(1) # (B, D)

        # vertical conv Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = conv(item_seq_emb).squeeze(3) # the last dimension is always 1
                conv_out = self.act_conv(conv_out)
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) # after pool, the last dimension is 1,so squeeze
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        out = torch.cat([out_v, out_h], 1)
        out = self.dropout(out)

        # fully-connected layer
        seq_output = self.fc(out)
        
        return seq_output
    
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
        best_kpi = -1
        for epoch in range(1, self.epochs + 1):
            self.train()

            total_loss = 0.
            sample_num = 0

            start_time = time.time()
            for seq, target, _ in train_loader:
                self.optimizer.zero_grad()
                seq = seq.to(self.device) # (B,max_len)
                target = target.to(self.device) # (B)

                seq_output = self.forward(seq) # (B, D)
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = F.cross_entropy(logits, target)

                reg_loss = self.item_embedding.weight.norm(2) + self.conv_v.weight.norm(2) + self.fc.weight.norm(2)
                loss = loss + self.reg_loss_conv_h() + self.reg_weight * reg_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                sample_num += target.numel()

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

    def predict(self, test_loader, k:list=[10]):
        self.eval()

        preds = {topk : torch.tensor([]) for topk in k}
        last_item = torch.tensor([])

        for seq, target, _ in test_loader:
            seq = seq.to(self.device)

            seq_output = self.forward(seq) 
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # (B, D), (N, D)->(B, N)

            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 

            for topk in k:
                preds[topk] = torch.cat((preds[topk], rank_list[:, :topk].cpu()), 0)
        
            last_item = torch.cat((last_item, target), 0)

        return preds, last_item

