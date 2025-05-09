'''
search space with Optuna 
'''
def define_space(trial, config):
    if config['model'] in ['gru4rec', 'sgru4rec']:
        config['item_embedding_dim'] = trial.suggest_int('item_embedding_dim', 40, 150, step=2)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['dropout_prob'] = trial.suggest_float('dropout_prob', 0.1, 0.9)
    elif config['model'] in ['sasrec', 'sfsrec']:
        config['item_embedding_dim'] = trial.suggest_int('item_embedding_dim', 40, 150, step=2)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        config['num_heads'] = trial.suggest_int('num_heads', 1, 2)
        config['dropout_prob'] = trial.suggest_float('dropout_prob', 0.1, 0.9)
    elif config['model'] in ['bsarec']:
        config['item_embedding_dim'] = trial.suggest_int('item_embedding_dim', 40, 150, step=2)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        config['num_heads'] = trial.suggest_int('num_heads', 1, 2)
        config['dropout_prob'] = trial.suggest_float('dropout_prob', 0.1, 0.9)
    elif config['model'] in ['narm', 'lrurec', 'fmlp', 'ssr']:
        config['item_embedding_dim'] = trial.suggest_int('item_embedding_dim', 40, 150, step=2)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        config['dropout_prob'] = trial.suggest_float('dropout_prob', 0.1, 0.9)
    elif config['model'] in ['caser', 'scaser']:
        config['item_embedding_dim'] = trial.suggest_int('item_embedding_dim', 40, 150, step=2)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['reg_weight'] = trial.suggest_float('reg_weight', 1e-5, 1e-3, log=True)
        config['dropout_prob'] = trial.suggest_float('dropout_prob', 0.1, 0.9)
    elif config['model'] in ['srgnn', 'srsgnn']:
        config['item_embedding_dim'] = trial.suggest_int('item_embedding_dim', 40, 150, step=2)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['step'] = trial.suggest_int('step', 1, 3)
    else:
        raise NotImplementedError('No need to tune hyper parameters...')
    
    return config