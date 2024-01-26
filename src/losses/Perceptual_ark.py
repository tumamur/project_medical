from src.models.ARK import ARKModel
from torch import nn
import torch
import torch.nn as nn


class ArkPerceptualLoss(nn.Module):
    def __init__(self):
        super(ArkPerceptualLoss, self).__init__()
        self.model = ARKModel(13, ark_pretrained_path='/home/max/Desktop/MLMI/Ark/ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar-20231123T004841Z-001/ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar')


class ModifiedSwinTransformer(nn.Module):
    def __init__(self, original_model):
        super(ModifiedSwinTransformer, self).__init__()
        # Assuming original_model is the ARKModel instance that contains the SwinTransformer
        self.model = original_model.model  # Access the SwinTransformer model directly

        # Define layers to extract features from
        self.selected_layers = [
            '0',  # First block of the first BasicLayer
            '1',  # First block of the second BasicLayer
            '2',  # First block of the third BasicLayer
            '3',  # First block of the fourth BasicLayer
        ]

    def forward(self, x):
        features = []

        # Step 1: Use the patch embedding layer
        x = self.model.patch_embed(x)  # Transform [B, C, H, W] to [B, L, C']
        x = self.model.pos_drop(x)
        # print(self.model.layers)
        # Step 2: Forward pass through selected layers
        for layer_idx, layer in enumerate(self.model.layers):
            x = layer(x)
            if str(layer_idx) in self.selected_layers:  # Assuming selected_layers are indices as strings
                # Optionally, reshape x back to [B, C, H, W] format if needed for your application
                features.append(x)

        return features


class PerceptualLossXray(nn.Module):
    def __init__(self):
        super(PerceptualLossXray, self).__init__()
        arkmodel = ArkPerceptualLoss()  # Initialize your ARK model
        print(arkmodel)
        self.modified_model = ModifiedSwinTransformer(arkmodel.model)  # Pass the Swin Transformer model

    def forward(self, images_real, images_cycle):
        features_orig = self.modified_model(images_real)
        features_gen = self.modified_model(images_cycle)

        loss = 0.0
        for f_orig, f_gen in zip(features_orig, features_gen):
            loss += torch.nn.functional.l1_loss(f_orig, f_gen)  # You can also use other loss functions

        return loss
