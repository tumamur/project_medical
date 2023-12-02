import pytorch_lightning as pl
import torch.nn as nn
import os
from src.modules.ChexpertModule import ChexpertDataModule
from src.utils.environment_settings import env_settings
from src.utils.utils import *
from src.utils._prepare_data import DataHandler
from src.models.image_to_classification_model import ArkModel
from src.models.image_to_classification_model import BioVILModel

params = read_config(env_settings.CONFIG)
processor = DataHandler(opt=params["dataset"])

chexpert_data_module = ChexpertDataModule(opt=params['dataset'], processor=processor)
chexpert_data_module.setup()

train_dataloader = chexpert_data_module.train_dataloader()
val_dataloader = chexpert_data_module.val_dataloader()
test_dataloader = chexpert_data_module.test_dataloader()

criterion = nn.BCEWithLogitsLoss()

# load model based on specification in the config.yaml
if params["model"]["image_encoder_model"] == "Ark":
    model = ArkModel(params["model"]["num_classes"], params["model"]["learning_rate"], criterion,
                     params["model"]["Ark_pretrained_path"])
    filename_base = "Ark_model_"
elif params["model"]["image_encoder_model"] == "BioVil":
    model = BioVILModel(params["model"]["num_classes"], 128, params["model"]["learning_rate"], criterion)
    filename_base = "biovil-t_"

# Define tensorboard logger
tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=env_settings.TENSORBOARD_LOG, version=1)

# Define ModelCheckpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=env_settings.MODEL_PATH,  # Specify the directory to save the models
    filename=os.path.join(filename_base + "{epoch:02d}_{val_loss:.4f}"),
    save_top_k=1,  # Save only the best model based on validation loss
    monitor='val_loss',
    mode='min',
)

trainer = pl.Trainer(max_epochs=params["model"]["n_epoch"], check_val_every_n_epoch=params["model"]["check_val_every_n_epochs"], logger=tensorboard_logger)
trainer.fit(model, train_dataloader, val_dataloader)

