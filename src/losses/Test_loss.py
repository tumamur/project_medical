import torch
import torch.nn as nn
import pandas as pd


class ClassificationLoss(nn.Module):
    def __init__(self, reference_path):
        super(ClassificationLoss, self).__init__()
        self.device = "cuda"
        df = pd.read_csv(reference_path)
        df = df.iloc[:, 6:].values
        self.data_ref = torch.tensor(df, dtype=torch.float32).to(self.device)
        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, output):
        # print(output)
        similarity = self.similarity(output, self.data_ref)
        # print(similarity)
        max_similarity, _ = similarity.max(dim=0)
        # print(max_similarity)
        loss = 1 - max_similarity.mean()

        return loss


# reference_path = '/home/max/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv'
# criterion = TestLoss(reference_path)
# output = torch.randint(2, size=(1, 14))
# print(output)

# loss = criterion(output)
# print(loss)
