import torch
import torch.nn as nn
from utils.post_processing import *

class CombinationLoss(nn.Module):
    def __init__(self, loss_function, data_imputation='zeros', threshold=0.5):
        super().__init__()
        self.loss_function = loss_function
        self.data_imputation = data_imputation
        self.threshold = threshold

    def forward(self, n_batch_distribution_tensor):
        # Adding a small epsilon to avoid log(0)

        # post_process returns a list of tensors, each tensor is a batch of probabilities
        # Each tensor is a batch of probabilities for each disease combination
        # Each tensor is of shape (batch_size, num_combinations)
        # We need to calculate the loss for each tensor in the list, and then average them
        # The loss for each tensor is the loss between the tensor and the reference tensor
        # The reference tensor is the tensor of probabilities for each disease combination in the entire dataset
        # The reference tensor is of shape (num_combinations)

        target_tensor, ref_tensor = post_process(n_batch_distribution_tensor, self.data_imputation, self.threshold)
        if self.loss_function == 'KLDivergence':
            return self.KL_divergence(target_tensor, ref_tensor)
        elif self.loss_function == 'CosineSimilarity':
            return self.cosine_similarity(target_tensor, ref_tensor)
        elif self.loss_function == 'Wasserstein':
            return self.wasserstein(target_tensor, ref_tensor)
        elif self.loss_function == 'JSDivergence':
            return self.js_divergence(target_tensor, ref_tensor)
        elif self.loss_function == 'Occurences':
            return self.occurences(target_tensor, ref_tensor)
        elif self.loss_function == 'MSE':
            return self.mse(target_tensor, ref_tensor)
        else:
            raise ValueError('Invalid loss function name: {}'.format(self.loss_function))

    def KL_divergence(self, avg_tensor, ref_tensor):
        # Adding a small epsilon to avoid log(0)
        epsilon = 1e-10
        avg_tensor = avg_tensor + epsilon
        ref_tensor = ref_tensor + epsilon
        return torch.sum(ref_tensor * torch.log(ref_tensor / avg_tensor))

    def cosine_similarity(self, avg_tensor, ref_tensor):
        cosine_similarity = torch.nn.functional.cosine_similarity(avg_tensor.unsqueeze(0), ref_tensor.unsqueeze(0), dim=1)
        # Since we want a loss, not a similarity, subtract from 1
        return 1 - cosine_similarity

    def wasserstein(self, avg_tensor, ref_tensor):
        return torch.mean(torch.abs(avg_tensor - ref_tensor))
    
    def js_divergence(self, avg_tensor, ref_tensor):
        # Calculate the average of the two distributions
        m = 0.5 * (avg_tensor + ref_tensor)
        # KL Divergence between m and each distribution
        kl_div_avg = self.KL_divergence(avg_tensor, m)
        kl_div_ref = self.KL_divergence(ref_tensor, m)
        # JS Divergence is the average of these two KL divergences
        js_div = 0.5 * (kl_div_avg + kl_div_ref)
        return js_div
    
    def occurences(self, avg_tensor, ref_tensor):
        # TODO : Implement this
        return
    
    def mse(self, avg_tensor, ref_tensor):
        return torch.mean((avg_tensor - ref_tensor) ** 2)