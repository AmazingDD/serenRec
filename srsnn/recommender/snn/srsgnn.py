import time
import math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, functional, layer

class GNN(nn.Module):
    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = nn.Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

    def GNNCell(self, A, hidden):
        input_in = (
            torch.matmul(A[:, :, : A.size(1)], 
                         self.linear_edge_in(hidden)) + self.b_iah
        ) # (B, L, D)
        input_out = (
            torch.matmul(
                A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # [B, L, 2D]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [B, L, gate_size], gate_size=3D
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (B, L, 3D) -> 3* (B, L, D)
        i_r, i_i, i_n = gi.chunk(3, dim=2)
        h_r, h_i, h_n = gh.chunk(3, dim=2)
        # GRU for gated GNN
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy # (B, L, D)

    def forward(self, A, hidden):
        # step for gated mechanism
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SRSGNN(nn.Module):
    def __init__(self, item_num, params):
        super(SRSGNN, self).__init__()

        self.embedding_size = params["item_embedding_dim"]
        self.step = params["step"]
        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.T = params['T']

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = layer.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.linear_two = layer.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.linear_three = layer.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = layer.Linear(self.embedding_size * 2, self.embedding_size, bias=False)

        self.lif_q = neuron.LIFNode(tau=params['tau'], detach_reset=True)
        self.ln_q = nn.LayerNorm(self.embedding_size)

        self.lif_sh = neuron.LIFNode(tau=params['tau'], detach_reset=True)
        self.ln_sh = nn.LayerNorm(self.embedding_size * 2)

        functional.set_step_mode(self, 'm')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):
        mask = item_seq.gt(0)  # greater than zero means true items
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1) # L
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq: # for each sequence (L)
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0]) # [true_items..., fake items...]
            u_A = np.zeros((max_n_node, max_n_node)) # [L, L] # adjacent matrix for one session graph 

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0: # meet fake items, then stop
                    break

                u = np.where(node == u_input[i])[0][0] # start node for an edge
                v = np.where(node == u_input[i + 1])[0][0] # end node for an edge
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0) # 
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)  # edge-in matrix (L, L)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1 
            u_A_out = np.divide(u_A.transpose(), u_sum_out) # edge-out matrix (L, L)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose() # (L, 2L)
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [B, L]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [B, L, 2L]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [B, L]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask # [B, L]
    
    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]) # (B)->(B, 1, D)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def forward(self, item_seq, item_seq_len):
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = self.item_embedding(items) # (B, L, D)
        hidden = self.gnn(A, hidden)  # (B, L, D)

        alias_inputs = alias_inputs.unsqueeze(2).expand(-1, -1, self.embedding_size) # ->(B, L, 1) -> (B, L, D)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs) # (B, L, D) select hidden representation as the actual order
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1) # (B, D) 

        seq_hidden = seq_hidden.unsqueeze(0).repeat(self.T, 1, 1, 1) # (T, B, L, D) global embedding
        ht = ht.unsqueeze(0).repeat(self.T, 1, 1) # (T, B, D) local embedding
        mask = mask.unsqueeze(0).repeat(self.T, 1, 1) # (T, B, L)

        q1 = self.linear_one(ht).unsqueeze(2) # -> (T, B, D) -> (T, B, 1, D)
        q2 = self.linear_two(seq_hidden) # -> (T, B, L, D)  

        # spike attention
        # alpha = self.linear_three(torch.sigmoid(q1 + q2)) # (T, B, L, 1)
        Q = q1 + q2
        T, B, L, D = Q.shape
        Q = self.ln_q((Q).flatten(0, 1)).reshape(T, B, L, D).contiguous()
        Q = self.lif_q(Q)
        alpha = self.linear_three(Q) # (T, B, L, 1)

        a = torch.sum(alpha * seq_hidden * mask.unsqueeze(3).float(), 2) # (T, B, L, 1) * (T, B, L, D) * (T, B, L, 1)->(T, B, L, D)->(T, B, D)
        
        sh = torch.cat([a, ht], dim=2) # (T, B, 2D)
        sh = self.ln_sh(self.lif_sh(sh).flatten(0, 1)).reshape(T, B, 2 * D).contiguous()

        seq_output = self.linear_transform(sh) #  ->(T, B, D)

        return seq_output.mean(0) # (B, D)
    
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

                seq_output = self.forward(seq, lens)
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
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
    

