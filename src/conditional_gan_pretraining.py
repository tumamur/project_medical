import torch
import pytorch_lightning as pl
import os

# Import your CGANModule here
from modules.cGAN_Module import CGANModule  # Replace with the actual import

from modules.ChexpertModule import ChexpertDataModule
from utils.environment_settings import env_settings
from utils._prepare_data import DataHandler
from utils.utils import read_config

def main():
    params = read_config(env_settings.CONFIG)
    processor = DataHandler(opt=params["dataset"])

    chexpert_data_module = ChexpertDataModule(opt=params['dataset'], processor=processor)
    chexpert_data_module.setup()
    train_dataloader = chexpert_data_module.train_dataloader()
    val_dataloader = chexpert_data_module.val_dataloader()
    test_dataloader = chexpert_data_module.test_dataloader()

    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=env_settings.TENSORBOARD_LOG_IMAGE_PRETRAINING)

    filename_base = "cGAN_"
    # Initialize CGANModule
    model = CGANModule(num_classes=params["trainer"]["num_classes"], params=params)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=env_settings.IMAGE_MODEL_PATH,
        filename=os.path.join(filename_base + "{epoch:02d}_{val_loss:.4f}"),
        every_n_epochs=1
    )

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10, check_val_every_n_epoch=1,
                         logger=tensorboard_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)  # Pass the entire data module


if __name__ == '__main__':
    main()
