import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

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


# Define the Gram Matrix calculation
class GramMatrix(nn.Module):
    def forward(self, input):
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size * channels, height * width)
        gram_matrix = torch.mm(features, features.t())
        return gram_matrix.div(batch_size * channels * height * width)


# Define the Style Loss class
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

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
