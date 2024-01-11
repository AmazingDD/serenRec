import os
import yaml
import optuna
import argparse

import torch

from srsnn.utils import *
from srsnn.recommender.ann.conventions import MF, Pop
from srsnn.recommender.snn.conventions import SMF
from srsnn.recommender.ann.gru4rec import GRU4Rec
from srsnn.recommender.snn.sgru4rec import SGRU4Rec
from srsnn.recommender.ann.sasrec import SASRec
from srsnn.recommender.snn.sfsrec import SFSRec
from srsnn.recommender.ann.caser import Caser
from srsnn.recommender.snn.scaser import Scaser
from srsnn.recommender.ann.stamp import STAMP
from srsnn.recommender.snn.stsamp import STSAMP
from srsnn.recommender.ann.srgnn import SRGNN
from srsnn.recommender.snn.srsgnn import SRSGNN

config = yaml.safe_load(open('./srsnn/config/basic.yaml', 'r'))

parser = argparse.ArgumentParser(description='SNN for sequential recommendation experiments')

# enviroment settings
parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
parser.add_argument('-gpu_id', type=str, default='0', help='gpu card id')
parser.add_argument('-use_cuda', action='store_true', help='use gpu to run code')
parser.add_argument('-worker', default=4, type=int, help='number of workers for dataloader') # 0
parser.add_argument('-shuffle', action='store_false', help='Whether or not to shuffle the training data before each epoch.')
# Training Settings
parser.add_argument('-dataset', default='ml-1m', help='dataset name')
parser.add_argument('-act', default='ann', help='algo type name')
parser.add_argument('-model', default='mf', help='algo name')
parser.add_argument('-prepro', default='5core', help='preprocessing method for dataset') # raw TODO
parser.add_argument('-len', '--max_seq_len', default=20, type=int, help='max sequence length')
parser.add_argument('-test_ratio', default=0.2, type=float, help='test ratio for fold-out split')
parser.add_argument('-split_method', default='ufo', type=str, help='method for train-test split')
parser.add_argument('-epochs', default=20, type=int, help='The number of training epochs.')
parser.add_argument('-batch_size', default=128, type=int, help='batch size.')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
parser.add_argument('-tau', default=2., type=float, help='time constant of LIF neuron')
# alogo specific settings
parser.add_argument('-item_embedding_dim', default=64, type=int, help='embedding dimension for items')
parser.add_argument('-nl', '--num_layers', default=2, type=int, help='number of certain layers')
parser.add_argument('-step', default=1, type=int, help='number of step for GNN')
parser.add_argument('-nh', '--num_heads', default=2, type=int, help='number of heads for attention mechanism')
parser.add_argument('-dp', '--dropout_prob', default=0.3, type=float, help='probability for dropout layer')
parser.add_argument('-reg', '--reg_weight', default=1e-4, type=float, help='regularization weight.')

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
    if config['model'] == 'mf':
        model = MF(item_num, config)
    elif config['model'] == 'pop':
        model = Pop(item_num, config)
    elif config['model'] == 'gru4rec':
        model = GRU4Rec(item_num, config)
    elif config['model'] == 'sasrec':
        model = SASRec(item_num, config)
    elif config['model'] == 'caser':
        model = Caser(item_num, config)
    elif config['model'] == 'stamp':
        model = STAMP(item_num, config)
    elif config['model'] == 'srgnn':
        model = SRGNN(item_num, config)
    else:
        raise ValueError(f'Invalid model name: {config["model"]}')
elif config['act'] == 'snn':
    if config['model'] == 'mf':
        model = SMF(item_num, config)
    elif config['model'] == 'sgru4rec':
        model = SGRU4Rec(item_num, config)
    elif config['model'] == 'sfsrec':
        model = SFSRec(item_num, config)
    elif config['model'] == 'scaser':
        model = Scaser(item_num, config)
    elif config['model'] == 'stsamp':
        model = STSAMP(item_num, config)
    elif config['model'] == 'srsgnn':
        model = SRSGNN(item_num, config)
    else:
        raise ValueError(f'Invalid model name: {config["model"]}')
else:
    raise ValueError(f'Invalid activation name: {config["act"]}')

print('Start training...')
model.fit(train_dataloader, test_dataloader)
print('Finish training')

print('Reloading model with the best parameters for prediction performace')
model.load_state_dict(model.best_state_dict)
preds, last_item = model.predict(test_dataloader, k=config['topk']) # top10 default

print('The prediction results for test set is:')
for topk in config['topk']:
    pred = preds[topk]
    metrics = accuracy_calculator(pred, last_item)
    print('-----------------')
    for kpi in config['metrics']:
        print(f'{kpi}@{topk}: {metrics[kpi]:.4f}')
    print('-----------------')
