import torch.nn as nn
import torch
import torch.nn.functional as F

class ReportDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(ReportDiscriminator, self).__init__()

        self.output_shape = (1, )

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # Increased size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Added dropout for regularization
            
            nn.Linear(512, 256),  # Additional layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Additional dropout
            
            nn.Linear(256, 128),  # Additional layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Additional dropout
            
            nn.Linear(128, 1),  # Output layer
        )

    def forward(self, x):
        # Removed the sigmoid activation before the model as it's not usual to pre-activate inputs
        x = self.model(x)
        x = torch.sigmoid(x)  # Sigmoid at the output for binary classification
        return x




class VectorDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        """
        A discriminator model for binary vectors representing CheXpert labels.

        Parameters:
        - input_size (int): The size of the input vector (e.g., 14 for CheXpert labels).
        - hidden_sizes (list of int): Sizes of hidden layers.
        """
        super(VectorDiscriminator, self).__init__()
        self.output_shape = (1, )
        layers = []
        for i in range(len(hidden_sizes)):
            in_features = input_size if i == 0 else hidden_sizes[i-1]
            layers.append(nn.Linear(in_features, hidden_sizes[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
        
        # The final layer outputs a single value
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Parameters:
        - x (torch.Tensor): The input tensor (binary vector of CheXpert labels).

        Returns:
        - torch.Tensor: A single value representing the probability of x being real.
        """
        return self.model(x)

class CreativeDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(CreativeDiscriminator, self).__init__()
        self.output_shape = (1, )
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=-1),  # Ensure softmax is applied across features
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Apply attention: No need to transpose for 1D attention weights
        attention_weights = self.attention(features).squeeze(-1)  # Ensure it's [batch_size, feature_dim]
        context_vector = attention_weights * features  # Element-wise multiplication
        context_vector = torch.sum(context_vector, dim=1)  # Sum to combine features
        
        authenticity = self.classifier(context_vector.unsqueeze(-1))
        
        return authenticity



class ImageDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(ImageDiscriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_channels, out_channels, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # C64 -> C128 -> C256 -> C512
        self.model = nn.Sequential(
            *discriminator_block(channels, out_channels=64, normalize=False),
            *discriminator_block(64, out_channels=128),
            *discriminator_block(128, out_channels=256),
            *discriminator_block(256, out_channels=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        x = self.model(img)
        x = torch.sigmoid(x)
        return x
