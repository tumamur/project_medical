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


class Perceptual_xray(nn.Module):
    def __init__(self):
        super(Perceptual_xray, self).__init__()
        original_model = get_biovil_t_image_encoder().encoder
        self.feature_extractor = ModifiedResNet(original_model)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, y_true, y_pred):
        y_true_normalized = self.normalize(y_true / 255.)
        y_pred_normalized = self.normalize(y_pred / 255.)

        true_features = self.feature_extractor(y_true_normalized)
        pred_features = self.feature_extractor(y_pred_normalized)

        loss = nn.functional.mse_loss(true_features, pred_features)
        return loss





