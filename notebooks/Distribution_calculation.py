import pandas as pd
from itertools import combinations

dataset_path = '/home/max/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv'

df = pd.read_csv(dataset_path)

# Replace empty or -1 values with a marker for uncertainty or absence
# df.fillna(0, inplace=True)  # Replace NaN values with 'Unknown'
# df.replace(-1, 0, inplace=True)  # Replace -1 values with 'Unknown'


diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
            'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax']

# Initialize a dictionary to store counts for each disease combination
disease_combination_counts = {}
number = 0
# Iterate through each possible combination of diseases
for r in range(1, len(diseases) + 1):
    for combo in combinations(diseases, r):
        # Create a key for the combination
        key = ', '.join(combo)

        # Filter rows where the current combination of diseases is present and no other diseases
        filtered_df = df[(df[list(combo)].sum(axis=1) == r) & (df[diseases].sum(axis=1) == r)]

        # Count the occurrences of the current disease combination
        count_disease_combination = len(filtered_df)

        # Store the count in the dictionary
        if count_disease_combination > 0:
            disease_combination_counts[key] = count_disease_combination
            number += 1


print(f"Total number of occured combination options: {number}")
disease_df = pd.DataFrame(list(disease_combination_counts.items()), columns=['Combination', 'Number'])

disease_df.to_csv('/home/max/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_counts_zeros.csv', index=False)

