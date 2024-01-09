import yaml
import json
import numpy as np

def get_monitor_metrics_mode():
    return {
        'val_loss': 'min',
        'val_acc': 'max',
        'val_f1': 'max',
        'val_precision': 'max',
    }

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error reading the config file: {e}")
        return None
    
def read_disease_combination_counts(path):
    with open(path, 'r') as f:
        disease_combination_counts = json.load(f)
    return disease_combination_counts


def split_dataset(master_df, opt):
    """ 
    Add new label to the dataframe called "split" which will be used for train, val, test split
    """

    # Ensure random seed for reproducibility, if specified
    if 'random_seed' in opt:
        np.random.seed(opt['random_seed'])

    # Calculate dataset sizes
    test_size = int(len(master_df) * opt["dataset"]["test_size"])
    val_size = int(len(master_df) * opt["dataset"]["val_size"])

    # Randomly select rows for the test dataset
    test_indices = np.random.choice(master_df.index, size=test_size, replace=False)
    master_df.loc[test_indices, 'split'] = 'test'

    # Exclude test indices and randomly select rows for the validation dataset
    remaining_indices = master_df[~master_df.index.isin(test_indices)].index
    val_indices = np.random.choice(remaining_indices, size=val_size, replace=False)
    master_df.loc[val_indices, 'split'] = 'val'

    # The rest of the data will be training data
    train_indices = master_df[~master_df.index.isin(np.concatenate([test_indices, val_indices]))].index
    master_df.loc[train_indices, 'split'] = 'train'

    return master_df


