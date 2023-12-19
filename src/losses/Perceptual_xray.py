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
        # Initialize with the pre-trained model
        self.model = original_model
        print(self.model.named_children)
        # Define layers to extract features from
        self.selected_layers = [
            'layers.0.blocks.0',  # First block of the first BasicLayer
            'layers.1.blocks.0',  # First block of the second BasicLayer
            'layers.2.blocks.0',  # First block of the third BasicLayer
            'layers.3.blocks.0',  # First block of the fourth BasicLayer
        ]  # fill in with layer indices or names

    def forward(self, x):
        features = []
        for name, module in self.model.named_children():
            if name == 'model':  # Assuming 'model' is the SwinTransformer
                for layer_name, layer in module.named_children():
                    if layer_name.startswith('layers'):
                        for block_name, block in layer.named_children():
                            x = block(x)
                            full_name = f"{layer_name}.{block_name}"
                            if full_name in self.selected_layers:
                                features.append(x)
        return features

class PerceptualLossXray(nn.Module):
    def __init__(self):
        super(PerceptualLossXray, self).__init__()
        arkmodel = ArkPerceptualLoss()
        self.modified_model = ModifiedSwinTransformer(arkmodel)

    def forward(self, images_real, images_cycle):
        features_orig = self.modified_model(images_real)
        features_gen = self.modified_model(images_cycle)
        loss = 0.0
        for f_orig, f_gen in zip(features_orig, features_gen):
            loss += torch.nn.functional.l1_loss(f_orig, f_gen)  # or use L2 norm

        return loss

