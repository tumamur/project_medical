import os
import torch
from typing import Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class Chexpert(Dataset):
    def __init__(self, opt, processor, records, training=True) -> None:
        super().__init__()
        self.opt = opt
        self.processor = processor
        self.records = records
        self.training = training
        self.resize = (self.opt["input_size"], self.opt["input_size"])
        if self.training:
            self.transformations = self.get_train_transformations()
        else:
            self.transformations = self.get_test_transformations()

    def __len__(self) -> int:
        """ 
        Args:
            None
        Returns:
            int: length of dataset
        """
        return len(self.records)
    
    def __getitem__(self, index: int):
        """ 
        Args:
            index (int): index of data
        Returns:
            dict: data dictionary
        """
        record = self.records[index]

        chexpert_label = record["label"]
        # convert it to numpy array
        chexpert_label = np.array(chexpert_label)

        # get the image path
        img_path = record["img"]
        image = self.processor.load_image(img_path)
        image = self.transformations(image)
        return {"target": image, "report": chexpert_label}

    def get_train_transformations(self):
        # Basic transformations for training
        return transforms.Compose([
            transforms.Resize(self.resize),  # Resize images
            transforms.ToTensor(),  # Convert images to tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    
    def get_test_transformations(self):
        # Basic transformations for testing (can be the same as training)
        return transforms.Compose([
            transforms.Resize(self.resize),  # Resize images
            transforms.ToTensor(),  # Convert images to tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    
    
