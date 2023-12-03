import torch
import itertools
import json

import torch
import itertools
from utils.environment_settings import env_settings
import json
from utils.utils import read_disease_combination_counts

overall_probabilities = {
    "zeros" : read_disease_combination_counts(path=env_settings.OCCURENCE_PROBABILITIES['zeros']),
    "ones" : read_disease_combination_counts(path=env_settings.OCCURENCE_PROBABILITIES['ones'])
}

def post_process(accumulated_batches, data_imputation, diseases, threshold=0.5):
    # Convert the model output probabilities into binary presence (0 or 1) based on the threshold

    combinations_in_batches = []
    output_probabilities = {}
    reference_probabilities = {}
    
    for output in accumulated_batches:
        presence = (output > 0.5).int()
        # Generate disease combinations as strings based on model output
        output_combination = [diseases[idx] for idx in range(len(output)) if presence[idx] == 1]
        if not output_combination:
            output_combination = ['No Finding']
        output_combination = ', '.join(sorted(output_combination))
        combinations_in_batches.append(output_combination)

    for combo in combinations_in_batches:
        if combo in output_probabilities:
            output_probabilities[combo] += 1
        else:
            output_probabilities[combo] = 1

    for combo in output_probabilities:
        output_probabilities[combo] /= len(output_probabilities)

    
    reference_probabilities = {combo: overall_probabilities[data_imputation].get(combo, 0) for combo in output_probabilities.keys()}


    output_distribution = torch.tensor(list(output_probabilities.values()), dtype=torch.float32)
    reference_distribution = torch.tensor(list(reference_probabilities.values()), dtype=torch.float32)

    output = {
        'combinations' : output_probabilities,
        'distributions' : output_distribution
    }

    reference = {
        'combinations' : reference_probabilities,
        'distributions' : reference_distribution
    }
    
    
    return output, reference
        




