import torch
from torch import nn
import torchvision.transforms as transforms
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder



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
        # x = self.original_model.backbone_to_vit(x)
        # for block in self.original_model.vit_pooler.blocks:
            # x = block(x)
            # features.append(x)

        return features


class Perceptual_xray(nn.Module):
    def __init__(self):
        super(Perceptual_xray, self).__init__()
        original_model = get_biovil_t_image_encoder().encoder
        # print(original_model)
        self.feature_extractor = ModifiedMultiImageEncoder(original_model)
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)

        loss = 0.0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss += nn.functional.mse_loss(true_feat, pred_feat)

        return loss




