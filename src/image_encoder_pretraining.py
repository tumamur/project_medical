import pytorch_lightning as pl
import torch.nn as nn
import os
from modules.ChexpertModule import ChexpertDataModule
from utils.environment_settings import env_settings
from utils.utils import *
from utils._prepare_data import DataHandler
from models.image_to_classification_model import ArkModel
from models.image_to_classification_model import BioVILModel


def main():
    params = read_config(env_settings.CONFIG)
    processor = DataHandler(opt=params["dataset"])

    chexpert_data_module = ChexpertDataModule(opt=params['dataset'], processor=processor)
    chexpert_data_module.setup()

    train_dataloader = chexpert_data_module.train_dataloader()
    val_dataloader = chexpert_data_module.val_dataloader()
    test_dataloader = chexpert_data_module.test_dataloader()

    criterion = nn.BCEWithLogitsLoss()

    # load model based on specification in the config.yaml
    if params["report_generator"]["image_encoder_model"] == "Ark":
        model = ArkModel(params["model"]["num_classes"], params["model"]["learning_rate"], criterion,
                         params["model"]["Ark_pretrained_path"])
        filename_base = "Ark_model_"
    elif params["report_generator"]["image_encoder_model"] == "BioVil":
        model = BioVILModel(params["report_generator"]["embedding_size"],
                            params["trainer"]["num_classes"],
                            params["report_generator"]["classification_head_hidden1"],
                            params["report_generator"]["classification_head_hidden2"],
                            params["report_generator"]["dropout_prob"], 0.01, criterion, params)
        filename_base = "biovil-t_"

    # Define tensorboard logger
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=env_settings.TENSORBOARD_LOG)

    print(env_settings.REPORT_MODEL_PATH)
    print(os.path.isdir(env_settings.REPORT_MODEL_PATH))
    # Define ModelCheckpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=env_settings.REPORT_MODEL_PATH,  # Specify the directory to save the models
        filename=os.path.join(filename_base + "{epoch:02d}_{val_loss:.4f}"),
        every_n_epochs=1
    )

    # trainer = pl.Trainer(max_epochs=params["model"]["n_epoch"],
    # check_val_every_n_epoch=params["model"]["check_val_every_n_epochs"], logger=tensorboard_logger)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10, check_val_every_n_epoch=1,
                         logger=tensorboard_logger)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

