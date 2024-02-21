import time
import math
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class FilterLayer(nn.Module):
    def __init__(self, max_seq_length, hidden_size, hidden_dropout_prob=0.5):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, hidden_size, hidden_act='gelu', hidden_dropout_prob=0.5):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size * 4)
        assert isinstance(hidden_act, str), 'invalid hidden acivation'
        self.intermediate_act_fn = ACT2FN[hidden_act]


        self.dense_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states

class Layer(nn.Module):
    def __init__(self, max_seq_length, hidden_size, hidden_dropout_prob, hidden_act):
        super(Layer, self).__init__()
        self.filterlayer = FilterLayer(max_seq_length, hidden_size, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, hidden_act, hidden_dropout_prob)

    def forward(self, hidden_states):

        hidden_states = self.filterlayer(hidden_states)
        intermediate_output = self.intermediate(hidden_states)

        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, n_layers, max_seq_length, hidden_size, hidden_dropout_prob, hidden_act):
        super(Encoder, self).__init__()
        layer = Layer(max_seq_length, hidden_size, hidden_dropout_prob, hidden_act)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class FMLP(nn.Module):
    def __init__(self, item_num, params):
        super(FMLP, self).__init__()

        self.epochs = params['epochs']
        self.device = params['device'] if torch.cuda.is_available() else 'cpu'
        self.lr = params['learning_rate'] 
        self.wd = params['weight_decay'] 

        self.n_layers = params['num_layers'] # 2
        self.hidden_size = params['item_embedding_dim'] # 64
        self.hidden_dropout_prob = params['dropout_prob'] # 0.5
        self.max_seq_length = params['max_seq_len']

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.layernorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.item_encoder = Encoder(
            n_layers=self.n_layers,
            max_seq_length=self.max_seq_length,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob, 
            hidden_act='gelu', # relu
        )

        self.apply(self.init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]) # (B)->(B, 1, D)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def add_position_embedding(self, sequence): # [B, L]
        seq_length = sequence.size(1) 
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence) # [B, L]
        item_embeddings = self.item_embedding(sequence) # [B, L, D]
        position_embeddings = self.position_embedding(position_ids) # [B, L, D]
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.layernorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    
    def forward(self, item_seq, item_seq_len):
        sequence_emb = self.add_position_embedding(item_seq) # [B, L, D]
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            output_all_encoded_layers=True)
        
        output = item_encoded_layers[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        return output

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

        return preds, last_item
    

