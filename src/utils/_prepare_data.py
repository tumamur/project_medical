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


class   DataHandler:
    def __init__(self, opt, mode="train") -> None:
        self.opt = opt
        self.mode = mode
        self.data_imputation = self.opt["data_imputation"]
        self.master_df = pd.read_csv(env_settings.MASTER_LIST[self.data_imputation])
        self.paired = self.opt["paired"]

        if not self.opt["use_all_images"]:
            # reduce the number of images
            # get sample for train split %80 * num_images
            # get sample for val split %10 * num_images
            # get sample for test split %10 * num_images

            train_df = self.master_df[self.master_df["split"] == "train"]
            val_df = self.master_df[self.master_df["split"] == "val"]
            test_df = self.master_df[self.master_df["split"] == "test"]

            train_df = self.reduce_data(train_df, num_samples=int(self.opt["num_images"] * 0.8))
            val_df = self.reduce_data(val_df, num_samples=int(self.opt["num_images"] * 0.1))
            test_df = self.reduce_data(test_df, num_samples=int(self.opt["num_images"] * 0.1))

            self.master_df = pd.concat([train_df, val_df, test_df])
            # reset the index
            self.master_df = self.master_df.reset_index(drop=True)

        self.records = self.create_records()

    def reduce_data(self, df, num_samples=1000):
        return df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    def create_records(self):

        records = []
        if self.paired:
            images_df, label_df = self.create_paired_dataset()
            print("Created paired dataset.")
        else:
            if self.mode != "train":
                images_df, label_df = self.create_paired_dataset()
                print("Created paired dataset for inference")
            else:
                images_df, label_df = self.create_unpaired_dataset()
                print("Created unpaired dataset.")

        labels = self.get_labels(label_df)
        images = self.get_images(images_df)
        splits = self.get_split(images_df)
        
        for i in range(len(images)):

            image_path = images[i].split("files")[-1]
            image_path = env_settings.DATA + image_path 
            
            record = {
                "img" : image_path,
                "label" : labels[i],
                "split" : splits[i]
            }
            if self.mode == "train":
                if splits[i] == "train" or splits[i] == "val":
                    records.append(record)
            elif self.mode == "inference_on_val":
                if splits[i] == "val":
                    records.append(record)
            elif self.mode == "inference":
                if splits[i] == "test":
                    records.append(record)
            else:
                raise NotImplementedError

        return records

    def create_unpaired_dataset(self):

        val_master_df = self.master_df[self.master_df["split"] == "val"]
        train_master_df = self.master_df[self.master_df["split"] == "train"]

        # shuffle the train and val dataframes
        print('length of train images:', len(train_master_df))
        print('length of val images:', len(val_master_df))

        print('shuffling train images to')
        train_images_df = train_master_df[['jpg', 'split']]
        train_labels_df = train_master_df[self.opt["chexpert_labels"]]

        train_images_df = train_images_df.sample(frac=1).reset_index(drop=True)

        val_images_df = val_master_df[['jpg', 'split']]
        val_labels_df = val_master_df[self.opt["chexpert_labels"]]

        # now create new images_df and label_df
        images_df = pd.concat([train_images_df, val_images_df])
        label_df = pd.concat([train_labels_df, val_labels_df])

        # reset the index
        images_df = images_df.reset_index(drop=True)
        label_df = label_df.reset_index(drop=True)
        
        return images_df, label_df

    def create_paired_dataset(self):

        label_df = self.master_df[self.opt["chexpert_labels"]]
        images_df = self.master_df[["jpg", "split"]]

        return images_df, label_df

    def get_images(self, images_df: pd.DataFrame):
        return images_df["jpg"].values.tolist()
    
    def get_split(self, images_df: pd.DataFrame):
        return images_df["split"].values.tolist()

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

    

