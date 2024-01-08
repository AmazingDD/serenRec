import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate, functional, layer

class SGRU4Rec(nn.Module):
    def __init__(self, item_num, params):
        super(SGRU4Rec, self).__init__()

        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.T = params['T']

        self.embedding_size = params['item_embedding_dim']
        self.hidden_size = params['item_embedding_dim'] # params['hidden_size']
        self.dropout_prob = params['dropout_prob']
        self.num_layers = 1 # params['num_layers']

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.lif = neuron.LIFNode(tau=params['tau'], v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.dense = layer.Linear(self.embedding_size, self.embedding_size)
        self.ln = nn.LayerNorm(self.embedding_size)

        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        functional.set_step_mode(self, 'm')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        B, L = item_seq.shape
        item_seq_emb = self.item_embedding(item_seq)  # (B, L, D)
        item_seq_emb = item_seq_emb.unsqueeze(0).repeat(self.T, 1, 1, 1) # -> (T, B, L, D)
        item_seq_emb = self.ln(item_seq_emb.flatten(0, 1)).reshape(self.T, B, L, -1).contiguous()
        snn_output = self.lif(item_seq_emb) # (T, B, L, D)

        gru_output, _ = self.gru_layers(snn_output.flatten(0, 1)) # -> (TB, L, D)
        gru_output = self.dense(gru_output.reshape(self.T, B, L, -1)) # ->(T, B, L, D)
        gru_output = gru_output.mean(0) # -> (B, L, D)

        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)

        return seq_output # ->(B, D)
    
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

                seq_output = self.forward(seq, lens) # (B, D)
                test_item_emb = self.item_embedding.weight # (N, D)
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # (B, N)
                loss = F.cross_entropy(logits, target)

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

    def predict(self, test_loader, k:list=[10]):
        self.eval()

        preds = {topk : torch.tensor([]) for topk in k}
        last_item = torch.tensor([])

        for seq, target, lens in test_loader:
            seq = seq.to(self.device)
            lens = lens.to(self.device)

            seq_output = self.forward(seq, lens) 
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # (B, D), (N, D)->(B, N)

            rank_list = torch.argsort(scores[:, 1:], descending=True) + 1 

            for topk in k:
                preds[topk] = torch.cat((preds[topk], rank_list[:, :topk].cpu()), 0)
        
            last_item = torch.cat((last_item, target), 0)

            functional.reset_net(self)

        return preds, last_item

