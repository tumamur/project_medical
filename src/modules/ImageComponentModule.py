import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from models.BioViL import BioViL
from models.ARK import ARKModel
from models.Discriminator import ReportDiscriminator
from losses.Metrics import Metrics
from torch.optim import lr_scheduler
from utils.environment_settings import env_settings
import torchmetrics.functional as F
from losses.ClassificationLoss import SimilarityLoss, AdversarialLoss

torch.autograd.set_detect_anomaly(True)


class ImageComponentModule(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.num_classes = opt['model']["num_classes"]
        self.embedding_size = opt['model']["embedding_size"]
        self.learning_rate = opt['model']["learning_rate"]
        self.loss_func = opt['model']["loss"]
        self.metrics = opt['model']["metrics"]
        self.optimizer_name = opt['model']["optimizer"]
        self.scheduler_name = opt['model']["scheduler"]
        self.weight_decay = opt['model']["weight_decay"]
        self.n_epoch = opt['model']["n_epoch"]
        self.accumulated_steps = opt['model']["accumulated_batches"]
        self.threshold = opt['model']["threshold"]
        self.data_imputation = opt['dataset']["data_imputation"]
        self.diseases = opt['dataset']['chexpert_labels']
        self.hidden_dim1 = opt['model']["classification_head_hidden1"]
        self.hidden_dim2 = opt['model']["classification_head_hidden2"]
        self.dropout_rate = opt['model']["dropout_prob"]
        self.accumulated_outputs = []
        self.image_encoder_model = opt['model']["image_encoder_model"]
        self.metric = Metrics(self.metrics, self.data_imputation, self.diseases, self.threshold)
        self.ref_path = env_settings.MASTER_LIST['zeros']
        self.criterion = self._get_criterion()
        self.model = self._get_model()
        self.discriminator = self._get_discriminator()
        self.batch_size = opt['dataset']['batch_size']

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        return x

    def reset_accumulation(self):
        self.accumulated_outputs.pop(0)

    def accumulate_outputs(self, outputs):
        self.accumulated_outputs.append(outputs)

    def compute_accumulated_metric(self):
        if not self.accumulated_outputs:
            # No previous data to compare with
            print("No previous data to compare with")
            return None
        accumulated_outputs = torch.cat(self.accumulated_outputs, dim=0)
        # Compute metric between current and previous batches
        metric = self.metric(accumulated_outputs)
        self.reset_accumulation()
        return metric

    def configure_optimizers(self):
        optimizer_dict = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            'D_Adam' : torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            'D_AdamW' : torch.optim.AdamW(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
        }
        if self.loss_func == "Adversarial":
            gen_optimizer = optimizer_dict[self.optimizer_name]
            d_optimizer_name = self.optimizer_name.replace('Adam', 'D_Adam')
            disc_optimizer = optimizer_dict[d_optimizer_name]
            return [gen_optimizer, disc_optimizer], [] 
        
        elif self.loss_func == "Similarity":
            optimizer = optimizer_dict[self.optimizer_name]
            return optimizer, [] 

        # lr_scheduler = self.init_lr_scheduler(self.scheduler_name, optimizer)
        # if lr_scheduler is not None:
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": lr_scheduler,
        #             "monitor": "val_loss",
        #         },
        #     }
        # return {"optimizer": optimizer}
        
    def discriminator_step(self, real_labels, fake_labels):
        # Ensure labels are of type Float
        real_labels = real_labels.float()

        # Adjust the shapes of real and fake labels for loss computation
        # Assuming real_labels is of shape [batch_size, num_classes]
        real_loss = self.criterion(self.discriminator(real_labels), torch.ones_like(real_labels))
        fake_loss = self.criterion(self.discriminator(fake_labels.detach()), torch.zeros_like(real_labels))
        disc_loss = (real_loss + fake_loss) / 2
        self.log('discriminator_loss', disc_loss, on_step=True, on_epoch=True)
        return disc_loss


    def generator_step(self, fake_labels):
        # Ensure fake_labels is of type Float
        fake_labels = fake_labels.float()
        # Compute generator loss
        gen_loss = self.criterion(self.discriminator(fake_labels), torch.ones_like(fake_labels))
        self.log('generator_loss', gen_loss, on_step=True, on_epoch=True)
        return gen_loss

    def training_step(self, train_batch, batch_idx):
        x = train_batch['target']
        y = train_batch['report']

        output = self.forward(x)
        output = torch.sigmoid(output) # fake labels
        self.accumulate_outputs(output)

        if self.loss_func == "Adversarial":
            disc_loss = self.discriminator_step(y, output)
            gen_loss = self.generator_step(output)
            train_loss = gen_loss

        elif self.loss_func == "Similarity":
            similarity_loss = self.criterion(output)
            train_loss = similarity_loss
            self.log('similarity_loss', similarity_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log('loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        if len(self.accumulated_outputs) == self.accumulated_steps:
            metric = self.compute_accumulated_metric()
            self.log('classification_metric', metric, on_step=True, on_epoch=True)

        return train_loss
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['target']
        y = val_batch['report']

        output = self.forward(x)
        output = torch.sigmoid(output)

        self.accumulate_outputs(output)

        if self.loss_func == "Adversarial":
            # For adversarial validation, compute loss for both generator and discriminator
            disc_loss = self.discriminator_step(y, output)
            gen_loss = self.generator_step(output)
            val_loss = gen_loss  # You may choose to report generator loss or a combination

        elif self.loss_func == "Similarity":
            # For similarity validation, compute similarity loss
            similarity_loss = self.criterion(output)
            val_loss = similarity_loss
            self.log('similarity_loss', similarity_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log('val_loss', similarity_loss, on_step=True, on_epoch=True, prog_bar=True)
        if len(self.accumulated_outputs) == self.accumulated_steps:
            metric = self.compute_accumulated_metric()
            self.log('classification_metric', metric, on_step=True, on_epoch=True)

        return val_loss
    
    def _get_criterion(self):
        if self.loss_func == "Similarity":
            return SimilarityLoss(self.ref_path)
        elif self.loss_func == "Adversarial":
            return AdversarialLoss()

    def _get_model(self):
        if self.image_encoder_model == "Ark":
            return ARKModel(num_classes=self.num_classes, ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK'])
        elif self.image_encoder_model == "BioVil":
            return BioViL(embedding_size=self.embedding_size, num_classes=self.num_classes, hidden_1=self.hidden_dim1,
                          hidden_2=self.hidden_dim2, dropout_rate=self.dropout_rate)
        
    def _get_discriminator(self):
        return ReportDiscriminator(input_dim=self.num_classes, output_dim=self.num_classes)

    # def init_lr_scheduler(self, name, optimizer):
    #     scheduler_dict = {
    #         "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch, eta_min=1e-7),
    #         "exponential": lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95),
    #         "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
    #                                                             verbose=True),
    #     }
    #     if name in scheduler_dict:
    #         return scheduler_dict[name]
    #     return None