import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, functional, layer

class STSAMP(nn.Module):
    def __init__(self, item_num, params):
        super(STSAMP, self).__init__()

        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.T = params['T']

        self.embedding_size = params["item_embedding_dim"]

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        self.w1 = layer.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = layer.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = layer.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = layer.Linear(self.embedding_size, 1, bias=False)

        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.embedding_size)
        self.ln3 = nn.LayerNorm(self.embedding_size)

        self.lif1 = neuron.LIFNode(tau=params['tau'], detach_reset=True)
        self.lif2 = neuron.LIFNode(tau=params['tau'], detach_reset=True)
        self.lif3 = neuron.LIFNode(tau=params['tau'], detach_reset=True)
        self.lif0 = neuron.LIFNode(tau=params['tau'], detach_reset=True)

        self.mlp_a = layer.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.mlp_b = layer.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.tanh = nn.Tanh()

        functional.set_step_mode(self, 'm')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]) # (B)->(B, 1, D)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq) 

        last_inputs = self.gather_indexes(item_seq_emb, item_seq_len - 1) # (B, D)
        org_memory = item_seq_emb # (B, L, D)

        org_memory = org_memory.unsqueeze(0).repeat(self.T, 1, 1, 1) # -> (T, B, L, D)
        last_inputs = last_inputs.unsqueeze(0).repeat(self.T, 1, 1) # -> (T, B, D)

        ms = torch.div(torch.sum(org_memory, dim=2), item_seq_len.unsqueeze(1).float().repeat(self.T, 1, 1)) # (T, B, D)

        alpha = self.count_alpha(org_memory, last_inputs, ms) # (T, B, L)

        vec = torch.matmul(alpha.unsqueeze(2), org_memory) # (T, B, 1, L) * (T, B, L, D) -> (T, B, 1, D)
        ma = vec.squeeze(2) + ms # ->(T, B, D)
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        seq_output = hs * ht # (T, B, D)
        return seq_output.mean(0) # (B, D)
    
    def count_alpha(self, context, aspect, output):
        ''' count the attention weights '''
        T, B, L, D = context.shape

        aspect_4dim = aspect.repeat(1, 1, L).reshape(T, B, L, D)
        output_4dim = output.repeat(1, 1, L).reshape(T, B, L, D)
        
        # all (T, B, L, D)
        res_ctx = self.w1(context) 
        res_asp = self.w2(aspect_4dim)
        res_output = self.w3(output_4dim)

        K = self.lif1(self.ln1(res_ctx.flatten(0, 1)).reshape(T, B, L, D).contiguous())
        Q = self.lif2(self.ln2(res_asp.flatten(0, 1)).reshape(T, B, L, D).contiguous())
        V = self.lif3(self.ln3(res_output.flatten(0, 1)).reshape(T, B, L, D).contiguous())

        res_sum = K + Q + V # (T, B, L, D)
        res_act = self.w0(res_sum)  # ->(T, B, L, 1)
        alpha = res_act.squeeze(3) # -> (T, B, L)
        spike_alpha = self.lif0(alpha) 

        return spike_alpha # (T, B, L)
    
    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)

        self.best_state_dict = None
        best_kpi = -1
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

                functional.reset_net(self)

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
