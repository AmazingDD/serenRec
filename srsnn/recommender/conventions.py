import torch
import torch.nn as nn

class BPRMF(nn.Module):
    def __init__(self, item_num, params, logger):
        super(BPRMF, self).__init__()
        self.n_factors = params['item_embedding_dim']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.logger = logger

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.n_factors, padding_idx=0)
        self.loss_func = BPRLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])
        # # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
                
    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        seq = seq.transpose(0,1)
        item_seq_emb = self.item_embedding(seq)
        org_memory = item_seq_emb
        uF = torch.div(torch.sum(org_memory, dim=1), torch.FloatTensor(lengths).unsqueeze(1).to(self.device)) # [b, emb]
        item_embs = self.item_embedding(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(uF, item_embs.permute(1, 0))
        #item_scores = self.sf(scores)
        return scores


    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)
        self.logger.info('Start training...')
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []

            for _, (seq, target, lens) in enumerate(train_loader):
                seq = seq.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                logit = self.forward(seq, lens)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                total_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            s = ''
            if valid_loader:
                self.eval()
                val_loss = []
                with torch.no_grad():
                    for _, (seq, target, lens) in enumerate(valid_loader):
                        seq = seq.to(self.device)
                        target = target.to(self.device)
                        logit = self.forward(seq, lens)
                        logit_sampled = logit[:, target.view(-1)]
                        loss = self.loss_func(logit_sampled)
                        val_loss.append(loss.item())
                s = f'\tValidation Loss: {np.mean(val_loss):3f}'

            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {np.mean(total_loss):.3f}' + s)
            
    def predict(self, test_loader, k=15):
        self.eval()
        preds, last_item = torch.tensor([]), torch.tensor([])
        for _, (seq, target_item, lens) in enumerate(test_loader):
            scores = self.forward(seq.to(self.device), lens)
            rank_list = (torch.argsort(scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items

            preds = torch.cat((preds, rank_list.cpu()), 0)
            last_item = torch.cat((last_item, target_item), 0)

        return preds, last_item