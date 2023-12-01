import pytorch_lightning as pl
import torch.nn as nn
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

if params["image_encoder_model"] == "Ark":
    model = ArkModel(params["num_classes_img_encoder"], params["learning_rate"], criterion)
elif params["image_encoder_model"] == "BioVil":
    model = BioVILModel(params["num_classes_img_encoder"], 128, params["learning_rate"], criterion)

trainer = pl.Trainer(max_epochs=params["max_epochs"], check_val_every_n_epoch=params["check_val_every_n_epochs"])
trainer.fit(model, train_dataloader, val_dataloader)

