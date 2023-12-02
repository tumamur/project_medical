import torch
import itertools
import json

import torch
import itertools
from utils.environment_settings import env_settings
import json
from utils.utils import read_disease_combination_counts

diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
            'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax']


overall_probabilities = {
    "zeros" : read_disease_combination_counts(path=env_settings.OCCURENCE_PROBABILITIES['zeros']),
    "ones" : read_disease_combination_counts(path=env_settings.OCCURENCE_PROBABILITIES['ones'])
}

def post_process(labels_arr, data_imputation, threshold=0.5):
    # Convert the model output probabilities into binary presence (0 or 1) based on the threshold

    n_batch_distribution = []
    n_batch_distribution_tensor = []
    n_reference_distribution_tensor = []
    n_reference_distribution = []
    overall_distirbution = overall_probabilities[data_imputation]
    print(len(labels_arr))

    for label in labels_arr:
        presence = (label > threshold).int()
        # Generate disease combinations as strings based on model output
        combinations_in_batch = []
        for batch in presence:
            diseases_present = [diseases[idx] for idx, present in enumerate(batch) if present == 1]
            if not diseases_present:
                diseases_present = ['No Finding']
            disease_combination = ', '.join(sorted(diseases_present))
            combinations_in_batch.append(disease_combination)
    
        # Calculate occurrence probabilities for each combination in the batch
        batch_distribution = {}
        for combo in combinations_in_batch:
            if combo in batch_distribution:
                batch_distribution[combo] += 1
            else:
                batch_distribution[combo] = 1
        for combo in batch_distribution:
            batch_distribution[combo] /= len(combinations_in_batch)
    
        # Convert batch_distribution to tensor for loss calculation
        batch_distribution_tensor = torch.tensor(list(batch_distribution.values()), dtype=torch.float32)
        n_batch_distribution_tensor.append(batch_distribution_tensor)
        n_batch_distribution.append(batch_distribution)

        # Extract only the relevant combinations from the entire dataset distribution
        reference_distribution = {combo: overall_distirbution.get(combo, 0) for combo in batch_distribution.keys()}
        reference_distribution_tensor = torch.tensor(list(reference_distribution.values()), dtype=torch.float32)
        n_reference_distribution_tensor.append(reference_distribution_tensor)
        n_reference_distribution.append(reference_distribution)

    return n_batch_distribution_tensor, n_reference_distribution_tensor





