import torch
from torch import nn
import torchvision.transforms as transforms
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder


class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        self.features = nn.Sequential(
            # Copy all layers up to and including layer3
            *list(original_model.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x

class ModifiedMultiImageEncoder(nn.Module):
    def __init__(self, original_model):
        super(ModifiedMultiImageEncoder, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        # Extract features from ResNet layers
        x = self.original_model.encoder.conv1(x)
        x = self.original_model.encoder.bn1(x)
        x = self.original_model.encoder.relu(x)
        x = self.original_model.encoder.maxpool(x)

        features = []
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.original_model.encoder, layer_name)
            x = layer(x)
            features.append(x)

        # Extract features from ViT layers
        x = self.original_model.backbone_to_vit(x)
        for block in self.original_model.vit_pooler.blocks:
            x = block(x)
            features.append(x)

        return features

class Perceptual_xray(nn.Module):
    def __init__(self):
        super(Perceptual_xray, self).__init__()
        original_model = get_biovil_t_image_encoder().encoder
        # print(original_model)
        # self.feature_extractor = ModifiedResNet(original_model)
        self.feature_extractor = ModifiedMultiImageEncoder(original_model)
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, y_true, y_pred):
        #y_true_normalized = self.normalize(y_true / 255.)
        #y_pred_normalized = self.normalize(y_pred / 255.)
        y_true_normalized = y_true
        y_pred_normalized = y_pred
        true_features = self.feature_extractor(y_true_normalized)
        print(true_features)
        pred_features = self.feature_extractor(y_pred_normalized)
        print(pred_features)

        loss = nn.functional.mse_loss(true_features, pred_features)
        return loss





