from tqdm import tqdm 
from pqdm.processes import pqdm
import pandas as pd
import logging
import os
import random
from typing import Sequence
import sys
from PIL import Image
from .environment_settings import env_settings



class DataHandler:
    def __init__(self, opt)-> None :
        self.opt = opt
        self.data_imputation = self.opt["data_imputation"]
        self.master_df = pd.read_csv(env_settings.MASTER_LIST[self.data_imputation])
        self.paired = self.opt["paired"]
        self.records = self.create_records()
    
    def create_records(self):

        records = []
        if self.paired:
            images_df, label_df = self.create_paired_dataset()
            print("Created paired dataset.")
        else:
            images_df, label_df = self.create_unpaired_dataset()
            print("Created unpaired dataset.")

        labels = self.get_labels(label_df)
        images = self.get_images(images_df)

        
        for i in range(len(images)):

            image_path = images[i].split("files")[-1]
            image_path = env_settings.DATA + image_path 
            
            record = {
                "img" : image_path,
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

    def create_paired_dataset(self):

        label_df = self.master_df[self.opt["chexpert_labels"]]
        images_df = self.master_df[["jpg"]]

        return images_df, label_df

    def get_images(self, images_df: pd.DataFrame):
        return images_df["jpg"].values.tolist()

    
    def load_image(self, img_path: str):
        image = Image.open(img_path).convert('RGB')
        return image
        


    def get_labels(self, label_df: pd.DataFrame):
        return [self.label_to_vector(label_row) for index, label_row in label_df.iterrows()]

    def label_to_vector(self, labels_raw):
        # Assuming labels_raw is a pandas Series of labels
        # Convert raw labels to a vector (e.g., one-hot encoding, numerical encoding, etc.)
        # Implementation depends on how you want to encode the labels
        return labels_raw.values  # simplistic example, needs to be adjusted based on actual label format

    

