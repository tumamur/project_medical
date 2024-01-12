import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F



class ClassificationLoss(nn.Module):
    def __init__(self, reference_path):
        super(ClassificationLoss, self).__init__()
        self.device = "cuda"
        df = pd.read_csv(reference_path)
        df = df.iloc[:, 7:-2].values
        self.data_ref = torch.tensor(df, dtype=torch.float32).to(self.device)

    def forward(self, output):

        # Only works for a batch_size of 1
        # similarity = self.similarity(output, self.data_ref)
        # max_similarity, _ = similarity.max(dim=0)
        # loss = 1 - max_similarity.mean()

        # Ensure that both output and data_ref have the same number of features (14)
        assert output.size(1) == self.data_ref.size(1), "Number of features must match."
        # Reshape output to [batch_size, 1, num_features] for broadcasting
        output = output.unsqueeze(1)
        # Calculate cosine similarity using torch.cosine_similarity
        similarity = F.cosine_similarity(output, self.data_ref.unsqueeze(0), dim=2)
        # Compute the maximum similarity for each row in output
        max_similarity, _ = similarity.max(dim=1)
        # Compute the loss
        loss = 1 - max_similarity.mean()

        return loss


# reference_path = '/home/max/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv'
# criterion = TestLoss(reference_path)
# output = torch.randint(2, size=(1, 14))
# print(output)

# loss = criterion(output)
# print(loss)
