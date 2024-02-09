import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.modules import MultiTaskModel
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder

class BioViL_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = get_biovil_t_image_encoder()
        self.multi_task_classifier = MultiTaskModel(
            input_dim=512, 
            classifier_hidden_dim=256, 
            num_classes=13, 
            num_tasks=1  # Assuming num_classes = num_tasks for Chexpert
        )

    def forward(self, x):
        # Pass input through the image model, which includes the downstream classifier
        x = self.image_model(x).img_embedding
        x = self.multi_task_classifier(x)
        x = x.squeeze(1)  # This will have a shape of [5, 13]
        return x

# Assuming other relevant definitions and imports are in place
