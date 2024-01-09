import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from modules.ChexpertModule import ChexpertDataModule
from modules.CycleGANModule import CycleGAN
from tensorboard import program
from utils._prepare_data import DataHandler
from utils.environment_settings import env_settings
from utils.utils import read_config

class Eval:
    def __init__(self, params):
        params = self.params

    # load model from dict
    def load_model(self, model_path):
        model = CycleGAN.load_from_checkpoint(model_path)
        return model

    def get_processor(self):
        processor = DataHandler(opt=self.params["dataset"], mode="inference_on_val")
        return processor
    
    def get_data_loader(self, processor):
        chexpert_data_module = ChexpertDataModule(opt=self.params['dataset'], processor=processor)
        chexpert_data_module.setup()
        test_dataloader = chexpert_data_module.test_dataloader()
        return test_dataloader

    def evaluate(self):
        raise NotImplementedError
