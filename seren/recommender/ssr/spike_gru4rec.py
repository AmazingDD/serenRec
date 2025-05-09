import time
import math
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate, functional, layer

class SGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.new_state = nn.Linear(input_size + hidden_size, hidden_size, bias=False)

        self.spike1 = surrogate.Erf()
        self.spike2 = surrogate.ATan()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x): 
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        for i in range(x.size(1)):
            input_step = x[:, i, :] # -> (B, D)
            combined = torch.cat((input_step, h), dim=1) # -> (B, 2D)
            rt = self.reset_gate(combined) # -> (B, D)
            zt = self.update_gate(combined) # -> (B, D)

            reset_gate = self.spike1(rt)
            update_gate = self.spike1(zt)

            # ->(B, 2D) -> (B, D)
            nt = self.new_state(torch.cat((input_step, reset_gate * h), dim=1))
            new_state_candidate = self.spike2(nt)

            new_state = update_gate * h + (1 - update_gate) * new_state_candidate

            outputs.append(new_state.unsqueeze(1))
            h = new_state

        output_tensor = torch.cat(outputs, dim=1)

        return output_tensor, h


class SpikeGRU4Rec(nn.Module):
    def __init__(self, item_num, params):
        super(SpikeGRU4Rec, self).__init__()
        self.logger = params['logger']
        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 
        self.T = params['T']
        self.max_seq_len = params['max_seq_len']
        
        self.embedding_size = params['item_embedding_dim']
        self.hidden_size = params['item_embedding_dim'] # params['hidden_size']
        self.num_layers = 1 # params['num_layers']

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.lif = neuron.LIFNode(tau=params['tau'], v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.dense = layer.Linear(self.embedding_size, self.embedding_size)
        self.ln = nn.LayerNorm(self.embedding_size)
        
        self.gru_layers = SGRU(
            input_size=self.embedding_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers
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
        snn_output = self.lif(item_seq_emb).mean(0) # (T, B, L, D) -> (B, L, D)
        
        gru_output,  _ = self.gru_layers(snn_output)
        
        gru_output = self.dense(gru_output) # ->(B, L, D)

        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)

        return seq_output # ->(B, D)
    
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