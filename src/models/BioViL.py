import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.modules import MultiTaskModel
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # self.image_inference = health_multimodal.image.get_image_inference(ImageModelType.BIOVIL_T)
        self.model = get_biovil_t_image_encoder()
        # print(self.model)

    def forward(self, x):
        x = self.model.forward(x).img_embedding
        # Check if normalization is needed
        x = torch.nn.functional.normalize(x, dim=1)
        return x

    def freeze_encoder(self, n_layers=None):
        """
        Freezes the parameters of the first n_layers of the encoder.
        If n_layers is None, all layers are frozen.
        """
        layers = list(self.model.children())  # Assuming the model is sequentially ordered
        n_freeze = len(layers) if n_layers is None else n_layers

        for layer in layers[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, start_layer, end_layer=None):
        """
        Unfreezes layers in a specified range.
        
        Parameters:
        - start_layer: Index of the first layer to unfreeze.
        - end_layer: Index of the last layer to unfreeze. If None, only the start_layer will be unfrozen.
        """
        layers = list(self.model.children())  # Adjust this if model children are not direct layers
        
        # If end_layer is not specified, set it to start_layer to unfreeze only one layer
        if end_layer is None:
            end_layer = start_layer
        
        for layer in layers[start_layer:end_layer + 1]:
            for param in layer.parameters():
                param.requires_grad = True




class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size_1, dropout_rate):
        super(ClassificationHead, self).__init__()
        hidden_dim_1 = hidden_size_1
        dropout_prob = dropout_rate
        self.fc1 = nn.Linear(input_size, hidden_dim_1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        return x



class BioViL(nn.Module):
    def __init__(self, embedding_size, num_classes, hidden_1, dropout_rate):
        super(BioViL, self).__init__()
        self.VisionTransformer = VisionTransformer()
        self.ClassificationHead = ClassificationHead(input_size=embedding_size, num_classes=num_classes,
                                                     hidden_size_1=hidden_1,
                                                     dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.VisionTransformer(x)
        x = self.ClassificationHead(x)
        return x


class BioViL_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = get_biovil_t_image_encoder()
        self.multi_task_classifier = MultiTaskModel(
            input_dim=512, 
            classifier_hidden_dim=256, 
            num_classes=1, 
            num_tasks=13  # Assuming num_classes = num_tasks for Chexpert
        )

    def forward(self, x):
        # Pass input through the image model, which includes the downstream classifier
        x = self.image_model(x).img_embedding
        x = self.multi_task_classifier(x)
        x = x.squeeze(1)  # Specify dimension to squeeze
        return x