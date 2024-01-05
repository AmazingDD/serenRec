import os
import yaml
import optuna
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
parser.add_argument('-act', default='ann', help='algo type name')
parser.add_argument('-model', default='bprmf', help='algo name')
parser.add_argument('-prepro', default='5core', help='preprocessing method for dataset') # raw TODO
parser.add_argument('-len', '--max_seq_len', default=20, type=int, help='max sequence length')
parser.add_argument('-test_ratio', default=0.2, type=float, help='test ratio for fold-out split')
parser.add_argument('-split_method', default='ufo', type=str, help='method for train-test split')
parser.add_argument('-epochs', default=20, type=int, help='The number of training epochs.')
parser.add_argument('-batch_size', default=128, type=int, help='batch size.')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
parser.add_argument('-tau', default=4./3, type=float, help='time constant of LIF neuron')
# alogo specific settings
parser.add_argument('-item_embedding_dim', default=64, type=int, help='embedding dimension for items')

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
dataset_name = f'dataset_{config["prepro"]}.pt'
if os.path.exists(os.path.join(dataset_dir, dataset_name)) and config['save_dataset']:
    print(f'{config["dataset"]}-{config["prepro"]} dataset already exists')
    train_dataset, test_dataset, item_num = torch.load(
        os.path.join(dataset_dir, dataset_name), map_location='cpu')
else:
    inters = Interactions(config)
    inters.build()

    train_dataset = SequentialDataset(inters.train_data)
    test_dataset = SequentialDataset(inters.test_data)
    item_num = inters.item_num
    if config['save_dataset']:
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        torch.save([train_dataset, test_dataset, item_num], os.path.join(dataset_dir, dataset_name))

print(f'train samples: {len(train_dataset)}, test samples: {len(test_dataset)}, max seq length: {config["max_seq_len"]}')
print(f'total item number after {config["prepro"]} preprocessing: {item_num}')

train_dataloader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['worker'])
test_dataloader = get_dataloader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['worker'])

print(config)
if config['act'] == 'ann':
    model = ANNBPRMF(item_num, config)
elif config['act'] == 'snn':
    model = SNNBPRMF(item_num, config)
else:
    raise ValueError('Invalid activation name...')

print('Start training...')
model.fit(train_dataloader, test_dataloader)
print('Finish training')

model.load_state_dict(model.best_state_dict)
preds, last_item = model.predict(test_dataloader, k=config['topk']) # top10 default

print('The prediction results for test set is:')
for topk in config['topk']:
    pred = preds[topk]
    metrics = accuracy_calculator(pred, last_item)
    for kpi in config['metrics']:
        print(f'{kpi}@{topk}: {metrics[kpi]:.4f}')
    print('-----------------')
