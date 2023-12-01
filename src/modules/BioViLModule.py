import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from src.models.BioViL import BioViL
from src.losses.CombinationLoss import CombinationLoss
from torch.optim import lr_scheduler
import torchmetrics.functional as mF

torch.autograd.set_detect_anomaly(True)

class BioViLModule(pl.LightningModule):
    
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.num_classes = opt["num_classes"]
        self.embedding_size = opt["embedding_size"]
        self.learning_rate = opt["learning_rate"]
        self.loss_func = opt["loss"]
        self.optimizer_name = opt["optimizer"]
        self.scheduler_name = opt["scheduler"]
        self.weight_decay = opt["weight_decay"]
        self.n_epoch = opt["n_epoch"]
        self.accumulated_steps = opt["accumulated_batches"]
        self.threshold = opt["threshold"]
        self.previous_outputs = []
        self.previous_labels = []
        self.current_outputs = []
        self.current_labels = []

        self.criterion = CombinationLoss(self.loss_func)
        self.model = self._get_model()

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        return x
    
    def reset_accumulation(self):
        self.previous_outputs = self.current_outputs
        self.previous_labels = self.current_labels
        self.current_outputs = []
        self.current_labels = []

    def accumulate_outputs_labels(self, outputs, labels, is_current=True):
        if is_current:
            self.current_outputs.append(outputs)
            self.current_labels.append(labels)
        else:
            self.previous_outputs.append(outputs)
            self.previous_labels.append(labels)

    def compute_accumulated_loss(self):
        if not self.previous_outputs or not self.current_outputs:
            # No previous data to compare with
            print("No previous data to compare with")
            return None

        prev_outputs = torch.cat(self.previous_outputs, dim=0)
        prev_labels = torch.cat(self.previous_labels, dim=0)
        current_outputs = torch.cat(self.current_outputs, dim=0)
        current_labels = torch.cat(self.current_labels, dim=0)

        # Compute loss between current and previous batches
        loss = self.criterion(current_outputs, prev_labels)
        self.reset_accumulation()
        return loss

    
    def configure_optimizers(self):
        optimizer_dict = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
        }
        optimizer = optimizer_dict[self.optimizer_name]
        lr_scheduler = self.init_lr_scheduler(self.scheduler_name, optimizer)
        if lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                },
            }
        return {"optimizer": optimizer}
    

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        output = self.forward(x)
        self.accumulate_outputs_labels(output, labels, is_current=True)

        if (batch_idx + 1) % self.accumulated_steps == 0:
            loss = self.compute_accumulated_loss()
            if loss is not None:
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        output = self.forward(x)
        self.accumulate_outputs_labels(output, labels, is_current=True)

        if (batch_idx + 1) % self.accumulated_steps == 0:
            loss = self.compute_accumulated_loss()
            if loss is not None:
                self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                return loss




    
    def _get_model(self):
        return BioViL(embedding_size=self.embedding_size, num_classes=self.num_classes)
    

    def init_lr_scheduler(self, name, optimizer):
        scheduler_dict = {
            "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch, eta_min=1e-7),
            "exponential": lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95),
            "ReduceLROnPlateau" : lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            "polynomial": lr_scheduler.PolynomialLR(optimizer, total_iters=self.opt.total_iterations, power=0.9),
        }
        if name in scheduler_dict:
            return scheduler_dict[name]
        return None
