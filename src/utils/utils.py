import yaml
import json
import numpy as np
import torch

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

def convert_to_soft_labels(type,label, current_epoch):
    """
    Convert labels to soft labels
    """
    if type == 'fixed':
        'Fixed, slightly less extreme value for labels. For instance, positive labels could be set to a value like 0.9 instead of 1, and negative labels to 0.1 instead of 0.'
        fix_value = 0.9
        smooth_positive_labels = label * fix_value  # Positive labels become 0.9
        smooth_negative_labels = (1 - label) * (1 - fix_value)  # Negative labels become 0.1
        soft_label = smooth_positive_labels + smooth_negative_labels

    elif type == 'smooth':
        'adjusting the labels slightly towards a uniform distribution'
        alpha = 0.1
        soft_label = label * (1 - alpha) + alpha * 0.5

    elif type == 'gaussian':
        'Adding Gaussian noise to labels, you need to ensure that the noise keeps the labels within their intended range: 0 to 1.'
        positive_noise = torch.randn_like(label) * 0.1  # Adjust the scale as needed
        negative_noise = torch.randn_like(label) * 0.1  # Adjust the scale as needed

        # Add noise conditionally
        soft_label = torch.where(label > 0.5,
                                torch.clamp(label + positive_noise, 0.5, 1),
                                torch.clamp(label + negative_noise, 0, 0.5))

    elif type == 'dynamic':
        'gradually increasing the range or intensity of the noise as training progresses'
        noise_level = min(0.1 + 0.01 * current_epoch, 0.5) 

        noisy_labels = label + (torch.rand_like(label) - 0.5) * noise_level
        soft_label = torch.clamp(noisy_labels, 0, 1)

    else:
        raise ValueError(f'Unknown soft label type: {type}')

    return soft_label


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


