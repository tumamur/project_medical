import os
import torch
from typing import Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Chexpert(Dataset):
    def __init__(self, opt, processor, records, training=True) -> None:
        super().__init__()
        self.opt = opt
        self.processor = processor
        self.records = records
        self.training = training
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




        return {"target": None, "report": chexpert_label}

    

    def get_train_transformations(self):

        # TODO : 
        # Start with torch transformations later use BioVil-T transformations

        return None
    
    def get_test_transformations(self):

        return None