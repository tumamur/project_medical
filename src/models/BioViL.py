import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # self.image_inference = health_multimodal.image.get_image_inference(ImageModelType.BIOVIL_T)
        self.model = get_biovil_t_image_encoder()
        # print(self.model)

    def forward(self, x):
        x = self.model.forward(x).projected_global_embedding
        # Check if normalization is needed
        x = torch.nn.functional.normalize(x, dim=1)
        return x



class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x



class BioViL(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(BioViL, self).__init__()
        self.VisionTransformer = VisionTransformer()
        self.ClassificationHead = ClassificationHead(input_size=embedding_size, num_classes=num_classes)

    def forward(self, x):
        x = self.VisionTransformer(x)
        x = self.ClassificationHead(x)
        return x