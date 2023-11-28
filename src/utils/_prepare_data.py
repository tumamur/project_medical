from tqdm import tqdm 
from pqdm.processes import pqdm
import pandas as pd
import logging
import os
import random
from typing import Sequence
import sys

from .environment_settings import env_settings



class DataHandler:
    def __init__(self, opt)-> None :
        self.opt = opt
        self.data_imputation = self.opt["data_imputation"]
        self.master_df = pd.read_csv(env_settings.MASTER_LIST[self.data_imputation])
    
    def create_records(self):

        records = []
        images_df, label_df = self.create_unpaired_dataset()

        labels = self.get_labels(label_df)
        images = self.get_images(images_df)

        
        for i in range(len(images)):
            record = {
                "img" : images[i],
                "label" : labels[i]
            }
            records.append(record)
        return records

    def create_unpaired_dataset(self):

        unpaired_samples = {}

        label_df = self.master_df[self.opt["chexpert_labels"]]
        images_df = self.master_df[["jpg"]]

        # TODO :now create unpaired records:
        # Shuffle one of the df to break the pairing

        images_df = images_df.sample(frac=1).reset_index(drop=True)

        return images_df, label_df
    

    def get_images(self, images_df: pd.DataFrame):
        return images_df["jpg"].values.tolist()

    
    def load_image(self, max_shape: Sequence[int], img_path: str):
        return None


    def get_labels(self, label_df: pd.DataFrame):
        return [self.label_to_vector(label_row) for index, label_row in label_df.iterrows()]

    def label_to_vector(self, labels_raw):
        # Assuming labels_raw is a pandas Series of labels
        # Convert raw labels to a vector (e.g., one-hot encoding, numerical encoding, etc.)
        # Implementation depends on how you want to encode the labels
        return labels_raw.values  # simplistic example, needs to be adjusted based on actual label format

    
