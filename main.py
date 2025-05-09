import os
import yaml
import optuna
import argparse

import torch

from seren.utils import *
from seren.space import define_space
# load conventional models
from seren.recommender import *
# load SNN-based models
from seren.recommender.ssr import SSR, SRSGNN, Scaser, SpikeGRU4Rec, SFSRec

MODEL = {
    'pop': Pop,
    'caser': Caser,
    'srgnn': SRGNN,
    'gru4rec': GRU4Rec,
    'narm': NARM,
    'stamp': STAMP,
    'sasrec': SASRec,
    'fmlp': FMLP,
    'lrurec': LRURec,
    'bsarec': BSARec,
    # SNN
    'scaser': Scaser,
    'srsgnn': SRSGNN,
    'sfsrec': SFSRec,
    'sgru4rec': SpikeGRU4Rec,
    'ssr': SSR,
}

config = yaml.safe_load(open('./seren/config/basic.yaml', 'r'))

parser = argparse.ArgumentParser(description='SNN for sequential recommendation experiments')

# enviroment settings
parser.add_argument('-data_path', type=str, default='./dataset/', help='root dir of dataset')
parser.add_argument('-gpu_id', type=str, default='0', help='gpu card id')
parser.add_argument('-use_cuda', action='store_true', help='use gpu to run code')
parser.add_argument('-worker', default=4, type=int, help='number of workers for dataloader') # 0
parser.add_argument('-shuffle', action='store_false', help='Whether or not to shuffle the training data before each epoch.')
parser.add_argument('-tune', action='store_true', help='activate optuna TPE tuning method to get best results')
parser.add_argument('-nt', '--num_trials', default=20, type=int, help='number of trials for tuning')
# Training Settings
parser.add_argument('-dataset', default='ml-1m', help='dataset name')
parser.add_argument('-model', default='sasrec', help='algo name')
parser.add_argument('-prepro', default='5core', help='preprocessing method for dataset')
parser.add_argument('-len', '--max_seq_len', default=20, type=int, help='max sequence length')
parser.add_argument('-test_ratio', default=0.2, type=float, help='test ratio for fold-out split')
parser.add_argument('-split_method', default='ufo', type=str, help='method for train-test split')
parser.add_argument('-epochs', default=20, type=int, help='The number of training epochs.')
parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size.')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
parser.add_argument('-tau', default=2., type=float, help='time constant of LIF neuron')
# algo specific settings
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

# init logger
if config['tune']:
    logger = Logger(config, desc='tune_test')
else:
    logger = Logger(config, desc='test')
config['logger'] = logger

dataset_root_dir = config['data_path'] 
dataset_dir = os.path.join(dataset_root_dir, config['dataset']) 
ensure_dir(dataset_dir)
dataset_name = f'dataset_{config["prepro"]}_seq{config["max_seq_len"]}.pt'
if os.path.exists(os.path.join(dataset_dir, dataset_name)) and config['save_dataset']:
    logger.info(f'{config["dataset"]}-{config["prepro"]}-{config["max_seq_len"]} dataset already exists')
    train_dataset, test_dataset, item_num = torch.load(
        os.path.join(dataset_dir, dataset_name), map_location='cpu')
else:
    inters = Interactions(config)
    inters.build()

    torch.save(inters, os.path.join(dataset_dir, f'inters_{config["prepro"]}_seq{config["max_seq_len"]}.pt'))

    train_dataset = SequentialDataset(inters.train_data)
    test_dataset = SequentialDataset(inters.test_data)
    item_num = inters.item_num
    if config['save_dataset']:
        torch.save([train_dataset, test_dataset, item_num], os.path.join(dataset_dir, dataset_name))

logger.info(f'train samples: {len(train_dataset)}, test samples: {len(test_dataset)}, max seq length: {config["max_seq_len"]}')
logger.info(f'total item number after {config["prepro"]} preprocessing: {item_num}')

train_dataloader = get_dataloader(
    train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['worker'])
test_dataloader = get_dataloader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['worker'])

# record final config info
for k, v in config.items():
    logger.info(f'{k}={v}')
logger.info('=' * 40)

if config['tune']:
    logger.info('start tuning...')
    def objective(trial, conf):
        conf = define_space(trial, conf)
        model = MODEL[conf['model']](item_num, conf)
        model.fit(train_dataloader, test_dataloader)

        return model.best_kpi # MRR@10

    study_name = f'optuna_{config["dataset"]}_{config["prepro"]}_{config["model"]}_tuning'
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize', study_name=study_name)
    config['epochs'] = 20
    study.optimize(lambda trial: objective(trial, config), n_trials=config['num_trials'])

    best_params = study.best_params
    for p_name, p_value in best_params.items():
        config[p_name] = p_value

    logger.info('already find out the best hyper-parameter settings, finish tuning...')
    logger.info(f'Best settings: {str(best_params)}')

# model initialzation
config['epochs'] = args.epochs
model = MODEL[config['model']](item_num, config)

# calulate number of parameters for model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"{config['dataset']}, {config['model']}, number of params: {n_parameters / 1000} k")

# model fitting
logger.info('Start training...')
model.fit(train_dataloader, test_dataloader)
logger.info('Finish training')

# model evaluation
if args.model != 'pop':
    # save best param
    ensure_dir(config['checkpoint_dir'])
    torch.save(
        model.best_state_dict, os.path.join(config['checkpoint_dir'], f"{config['dataset']}_{config['prepro']}_{config['model']}_chkpt.pt"))

    # load best param
    logger.info('Reloading model with the best parameters for prediction performace')
    model.load_state_dict(model.best_state_dict)
    res_kpi = model.evaluate(test_dataloader, k=config['topk'])

    # preds, last_item = model.predict(test_dataloader, k=config['topk']) # top10 default
    # print('The prediction results for test set is:')
    # for topk in config['topk']:
    #     pred = preds[topk]
    #     metrics = accuracy_calculator(pred, last_item)
    #     print('-----------------')
    #     for kpi in config['metrics']:
    #         print(f'{kpi}@{topk}: {metrics[kpi]:.4f}')
    #     print('-----------------')
    logger.info(f'The {config["model"]} prediction results for {config["dataset"]}-{config["prepro"]} test set is:')
    for topk in config['topk']:
        kpis = res_kpi[topk]
        logger.info('-----------------')
        for kpi_name in config['metrics']:
            logger.info(f'{kpi_name}@{topk}: {kpis[kpi_name]:.4f}')
    logger.info('-----------------')
else:
    logger.info('There is no need for Pop model re-implementation with best hyperparameters')
