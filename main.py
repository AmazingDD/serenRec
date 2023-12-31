import os
import yaml
import argparse

import torch

from srsnn.utils import *
from srsnn.recommender.ann.conventions import BPRMF as ANNBPRMF
from srsnn.recommender.snn.conventions import BPRMF as SNNBPRMF

config = yaml.safe_load(open('./srsnn/config/basic.yaml', 'r'))

parser = argparse.ArgumentParser(description='SNN for sequential recommendation experiments')

# enviroment settings
parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
parser.add_argument('-gpu_id', type=str, default='0', help='gpu card id')
parser.add_argument('-use_cuda', action='store_true', help='use gpu to run code')
parser.add_argument('-worker', default=0, type=int, help='number of workers for dataloader')
parser.add_argument('-shuffle', action='store_false', help='Whether or not to shuffle the training data before each epoch.')
# Training Settings
parser.add_argument('-dataset', default='ml-1m', help='dataset name')
parser.add_argument('-prepro', default='raw', help='preprocessing method for dataset') # 10core TODO
parser.add_argument('-len', '--max_seq_len', default=10, type=int, help='max sequence length')
parser.add_argument('-test_ratio', default=0.2, type=float, help='test ratio for fold-out split')
parser.add_argument('-epochs', default=150, type=int, help='The number of training epochs.')
parser.add_argument('-batch_size', default=128, type=int, help='batch size.')
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('-T', default=10, type=int, help='simulating time-steps') # this should be the same as max_seq_len
# alogo specific settings
parser.add_argument('-item_embedding_dim', default=100, type=int, help='embedding dimension for items')

args = parser.parse_args()
config.update(vars(args))

if config['use_cuda'] and torch.cuda.is_available():
    config['device'] = f"cuda:{config['gpu_id']}"
else:
    config['device'] = 'cpu'

if config['reproducibility']:
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataset_root_dir = './dataset/'
dataset_dir = os.path.join(dataset_root_dir, config['dataset']) 
if os.path.exists(os.path.join(dataset_dir, 'dataset.pt')):
    train_dataset, test_dataset, item_num = torch.load(
        os.path.join(dataset_dir, 'dataset.pt'), map_location='cpu')
else:
    inters = Interactions(config)
    inters.build()

    train_dataset = SequentialDataset(inters.train_data)
    test_dataset = SequentialDataset(inters.test_data)
    item_num = inters.item_num
    if config['save_dataset']:
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        torch.save([train_dataset, test_dataset, item_num], os.path.join(dataset_dir, 'dataset.pt'))

train_dataloader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['worker'])
test_dataloader = get_dataloader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['worker'])


model = ANNBPRMF(item_num, config)
print('Start training...')
model.fit(train_dataloader)
print('Finish training')

preds, last_item = model.predict(test_dataloader, k=config['topk']) # top10 default

print('The prediction results is:')
for pred in preds:
    metrics, k = accuracy_calculator(pred, last_item, config['metrics'])
    for kpi in config['metrics']:
        print(f'{kpi}@{k}: {metrics[kpi]:.4f}')
