import time
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import surrogate, neuron, functional, layer

class MLP(nn.Module):
    ''' FeedForward in spikeformer '''
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = layer.Linear(in_features, hidden_features)
        self.fc1_ln = nn.LayerNorm(hidden_features)
        self.fc1_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.fc2_linear = layer.Linear(hidden_features, out_features)
        self.fc2_ln = nn.LayerNorm(out_features)
        self.fc2_lif = neuron.LIFNode(tau=2.0, detach_reset=True)  # remember multistep

        self.c_hidden = hidden_features
        # self.c_output = out_features

    def forward(self, x):
        T, B, L, H = x.shape
        x = self.fc1_linear(x)  # (T, B, L, H) -> (T, B, L, c_hidden)
        x = self.fc1_ln(x.flatten(0, 1)).reshape(T, B, L, self.c_hidden).contiguous()
        x = self.fc1_lif(x) # (T, B, L, c_hidden)

        x = self.fc2_linear(x) # -> (T, B, L, H)
        x = self.fc2_ln(x.flatten(0, 1)).reshape(T, B, L, H).contiguous()
        x = self.fc2_lif(x)

        return x # -> (T, B, L, H)
    
class SSA(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None):
        super().__init__()
        # dim is the hidden size H set in SFSRec
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale # 0.125
        self.q_linear = layer.Linear(dim, dim, bias=qkv_bias)
        self.q_ln = nn.LayerNorm(dim)
        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.k_linear = layer.Linear(dim, dim, bias=qkv_bias)
        self.k_ln = nn.LayerNorm(dim)
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.v_linear = layer.Linear(dim, dim, bias=qkv_bias)
        self.v_ln = nn.LayerNorm(dim)
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.attn_lif = neuron.LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(dim, dim)
        self.proj_ln = nn.LayerNorm(dim)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, L, H = x.shape

        x_for_qkv = x.clone() # T, B, L, H

        q_linear_out = self.q_linear(x_for_qkv)  # T, B, L, H -> T, B, L, H
        q_linear_out = self.q_ln(q_linear_out.flatten(0, 1)).reshape(T, B, L, H).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, L, self.num_heads, H // self.num_heads).permute(0, 1, 3, 2, 4).contiguous() # T, B, L, H -> T, B, L, head_num, head_dim -> T, B, head_num, L, head_dim 

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_ln(k_linear_out.flatten(0, 1)).reshape(T, B, L, H).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, L, self.num_heads, H // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_ln(v_linear_out.flatten(0, 1)).reshape(T, B, L, H).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, L, self.num_heads, H // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale # -> (T, B, head_num, L, L)
        x = attn @ v # -> (T, B, head_num, L, head_dim)
        x = x.transpose(2, 3).reshape(T, B, L, H).contiguous() # -> (T, B, L, head_num, head_dim) -> (T, B, L, H)
        x = self.attn_lif(x)

        x = self.proj_linear(x)
        x = self.proj_ln(x.flatten(0, 1)).reshape(T, B, L, H).contiguous()
        x = self.proj_lif(x)

        return x # (T, B, L, H)

class Block(nn.Module):
    ''' Spikeformer Block '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        # mlp_hidden_dim = int(dim * mlp_ratio) # inner_size
        mlp_hidden_dim = 256 # stay the same as SASRec, this is inner_size
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x) # (T, B, L, H)
        x = x + self.mlp(x) # (T, B, L, H)
        return x # (T, B, L, H)

class SFSRec(nn.Module):
    '''SpikeFormer for Sequential Recommendation'''
    def __init__(self, item_num, params):
        super(SFSRec, self).__init__()
        self.logger = params['logger']
        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.T = params['T']
        self.max_seq_length = params['max_seq_len']
        self.layer_norm_eps = 1e-12 # params["layer_norm_eps"]
        self.hidden_dropout_prob = params['dropout_prob']
        
        self.hidden_size = params['item_embedding_dim']
        self.n_layers = params['num_layers'] # 2 number of spikeformers
        self.n_heads = params['num_heads'] # 2 number of head in spikeformer SSA

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.head = nn.Linear(self.hidden_size, self.hidden_size)

        self.blocks = nn.ModuleList(
            [Block(dim=self.hidden_size, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=False, qk_scale=0.125) for _ in range(self.n_layers)]
        )

        functional.set_step_mode(self, 'm')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]) # (B)->(B, 1, D)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def forward_features(self, item_seq):
        # feature extrator
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb) # (B, L, H)

        input_emb = input_emb.unsqueeze(0).repeat(self.T, 1, 1, 1) # ->(T, B, L, H)
        x = self.rpe_lif(input_emb) 

        # start spikeformer parts
        for blk in self.blocks:
            x = blk(x)
        return x # (T, B, L, H) 

    def forward(self, item_seq, item_seq_len):
        x = self.forward_features(item_seq)
        x = x.mean(0) # (T, B, L, H) -> (B, L, H)
        x = self.head(x) # -> (B, L, H)
        output = self.gather_indexes(x, item_seq_len - 1) # (B, H)

        return output
    
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
                seq = seq.to(self.device) # (B, L)
                target = target.to(self.device) # (B)
                lens = lens.to(self.device) # (B)

                seq_output = self.forward(seq, lens)
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = F.cross_entropy(logits, target)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                sample_num += target.numel()

                functional.reset_net(self)

            self.scheduler.step()

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

            functional.reset_net(self)

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
            functional.reset_net(self)

        for topk in k:
            res[topk] = {kpi: r / batch_cnt for kpi, r in res[topk].items()}
        return res
        