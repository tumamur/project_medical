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

        resnet_features = []
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.original_model.encoder, layer_name)
            x = layer(x)
            resnet_features.append(x)

        print(x.shape)
        x = self.original_model.encoder.avgpool(x)
        print(x.shape)
        # x = self.original_model.encoder.fc(x)
        # Prepare for ViT input
        x = self.original_model.backbone_to_vit(x)  # Convert to correct dimension for ViT
        print(self.original_model.vit_pooler.blocks)
        vit_features = []
        for block in self.original_model.vit_pooler.blocks:
            x = x.squeeze(-1).squeeze(-1)  # added
            x = x.unsqueeze(1)  # added
            x = block(x, None)  # Pass None or appropriate value if your block expects more arguments
            vit_features.append(x)

        # Combine ResNet and ViT features
        features = resnet_features + vit_features

        return features

class Perceptual_xray(nn.Module):
    def __init__(self):
        super(Perceptual_xray, self).__init__()
        original_model = get_biovil_t_image_encoder().encoder
        print(original_model)
        self.feature_extractor = ModifiedMultiImageEncoder(original_model)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)

        loss = 0.0
        for true_feat, pred_feat in zip(true_features, pred_features):
            print(nn.functional.mse_loss(true_feat, pred_feat))
            loss += nn.functional.mse_loss(true_feat, pred_feat)

        return loss
