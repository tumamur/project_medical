import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from models.ARK import ARKModel


##########################################################################################
class ClassificationLoss(nn.Module):
    """Loss function for the classification task."""
    """
    Example:
        criterion = TestLoss(reference_path)
        output = torch.randint(2, size=(1, 14))
        loss = criterion(output)
    """

    def __init__(self, reference_path):
        super(ClassificationLoss, self).__init__()
        self.device = "cuda"
        df = pd.read_csv(reference_path)
        df = df.iloc[:, 7:-2].values
        self.data_ref = torch.tensor(df, dtype=torch.float32).to(self.device)

    def forward(self, output):

        # Only works for a batch_size of 1
        # similarity = self.similarity(output, self.data_ref)
        # max_similarity, _ = similarity.max(dim=0)
        # loss = 1 - max_similarity.mean()

        # Ensure that both output and data_ref have the same number of features (14)
        assert output.size(1) == self.data_ref.size(1), "Number of features must match."
        # Reshape output to [batch_size, 1, num_features] for broadcasting
        output = output.unsqueeze(1)
        # Calculate cosine similarity using torch.cosine_similarity
        similarity = F.cosine_similarity(output, self.data_ref.unsqueeze(0), dim=2)
        # Compute the maximum similarity for each row in output
        max_similarity, _ = similarity.max(dim=1)
        # Compute the loss
        loss = 1 - max_similarity.mean()
        return loss
##########################################################################################  

biovil_model_dict = { 
    "biovil_t": get_biovil_t_image_encoder,
    "biovil": get_biovil_image_encoder
}

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
        return features
    
class PerceptualLoss_BioVil(nn.Module):
    def __init__(self, base_model):
        super(PerceptualLoss_BioVil, self).__init__()
        original_model = biovil_model_dict[base_model]().encoder
        self.feature_extractor = ModifiedMultiImageEncoder(original_model)
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, y_true, y_pred):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        loss = 0.0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss += nn.functional.mse_loss(true_feat, pred_feat)

        return loss
##########################################################################################  

class ArkPerceptualLoss(nn.Module):
    def __init__(self, ark_pretrained_path):
        super(ArkPerceptualLoss, self).__init__()
        self.model = ARKModel(13, ark_pretrained_path)

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

class PerceptualLoss_ARK(nn.Module):
    def __init__(self, ark_pretrained_path):
        super(PerceptualLoss_ARK, self).__init__()
        arkmodel = ArkPerceptualLoss(ark_pretrained_path)
        print(arkmodel)
        self.modified_model = ModifiedSwinTransformer(arkmodel)

    def forward(self, images_real, images_cycle):
        features_orig = self.modified_model(images_real)
        features_gen = self.modified_model(images_cycle)
        loss = 0.0
        for f_orig, f_gen in zip(features_orig, features_gen):
            loss += torch.nn.functional.l1_loss(f_orig, f_gen)  # or use L2 norm

        return loss
########################################################################################## 
    

class PerceptualLoss_VGG(nn.Module):
    def __init__(self):
        super(PerceptualLoss_VGG, self).__init__()
        # Load pretrained models
        vgg19 = models.vgg19(pretrained=True)
        vgg19 = vgg19.features
        vgg19.eval()

        # Choose intermediate layers for feature extraction
        self.content_layers = nn.Sequential(*list(vgg19.children())[:35])

        # Freeze the parameters of VGG19
        for param in self.content_layers.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        # Normalize pixel values to be in the range [0, 1]
        y_true = y_true / 255.0
        y_pred = y_pred / 255.0
    
    
        # Apply normalization for VGG19
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        y_true_normalized = normalize(y_true)
        y_pred_normalized = normalize(y_pred)
    
        # Extract features from the target (y_true) and generated (y_pred) images
        true_features = self.content_layers(y_true_normalized)
        pred_features = self.content_layers(y_pred_normalized)
    
        # Calculate perceptual loss as the mean squared difference between features
        loss = nn.functional.mse_loss(true_features, pred_features)
    
        return loss
########################################################################################## 
    
# Define the Gram Matrix calculation
class GramMatrix(nn.Module):
    def forward(self, input):
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size * channels, height * width)
        gram_matrix = torch.mm(features, features.t())
        return gram_matrix.div(batch_size * channels * height * width)


# Define the Style Loss class
class StyleLoss_VGG(nn.Module):
    def __init__(self):
        super(StyleLoss_VGG, self).__init__()

        # Load pre-trained VGG19 model
        vgg19 = models.vgg19(pretrained=True)
        vgg19 = vgg19.features
        vgg19.eval()

        # Choose intermediate layers for style feature extraction
        self.style_layers = [0, 5, 10, 19, 28]

        # Extract features from the selected style layers
        self.style_features = nn.ModuleList([nn.Sequential(*list(vgg19.children())[:layer+1]) for layer in self.style_layers])

        # Freeze the parameters of VGG19
        for param in self.parameters():
            param.requires_grad = False

        # Gram Matrix calculation
        self.gram = GramMatrix()

    def forward(self, y_true, y_pred):
        # Normalize pixel values to be in the range [0, 1]
        y_true = y_true / 255.0
        y_pred = y_pred / 255.0

        # Extract style features from the target (y_true) and generated (y_pred) images
        true_styles = [self.gram(feature(y_true)) for feature in self.style_features]
        pred_styles = [self.gram(feature(y_pred)) for feature in self.style_features]

        # Calculate style loss as the mean squared difference between Gram matrices
        loss = 0.0
        for true_style, pred_style in zip(true_styles, pred_styles):
            loss += nn.functional.mse_loss(true_style, pred_style)

        return loss
########################################################################################## 
    
class Loss:
    """Base class for all loss functions."""
    """
    Example:
    loss_instance = Loss("classification", reference_path="path_to_reference.csv")
    """
    def __init__(self, loss_type, **kwargs):
        self.loss_type = loss_type
        self.loss_fn = self._get_loss_function(loss_type, **kwargs)

    def _get_loss_function(self, loss_type, **kwargs):
        if loss_type == "classification":
            return ClassificationLoss(**kwargs)
        elif loss_type == "biovil":
            return PerceptualLoss_BioVil(base_model="biovil", **kwargs)
        elif loss_type == "biovil_t":
            return PerceptualLoss_BioVil(base_model="biovil_t", **kwargs)
        elif loss_type == "ark":
            return PerceptualLoss_ARK(**kwargs)
        elif loss_type == "vgg":
            return PerceptualLoss_VGG(**kwargs)
        elif loss_type == "style_vgg":
            return StyleLoss_VGG(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)



