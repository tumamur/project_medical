import torch
import itertools
from environment_settings import env_settings
import json
from utils import read_disease_combination_counts

diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
            'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax']

occurence_probabilities_ones = read_disease_combination_counts(path=env_settings.OCCURENCE_PROBABILITIES_ONES)
occurence_probabilities_zeros = read_disease_combination_counts(path=env_settings.OCCURENCE_PROBABILITIES_ZEROS)

def post_process(type, labels, threshold=0.5):
    # Convert the model output probabilities into binary presence (0 or 1) based on the threshold
    if type != "reference":
        presence = (labels > threshold).int()
    else:
        presence = labels

    # Generate disease combinations as strings based on model output
    combinations = []
    for batch in presence:
        diseases_present = [diseases[idx] for idx, present in enumerate(batch) if present == 1]
        disease_combination = ', '.join(sorted(diseases_present))
        combinations.append(disease_combination)

    # Calculate occurrence probabilities for each combination in the batch
    batch_distribution = {}
    for combo in combinations:
        if combo in batch_distribution:
            batch_distribution[combo] += 1
        else:
            batch_distribution[combo] = 1
    for combo in batch_distribution:
        batch_distribution[combo] /= len(combinations)

    # Convert batch_distribution to tensor for loss calculation
    batch_distribution_tensor = torch.tensor(list(batch_distribution.values()))
    return batch_distribution_tensor