
import torch
import torch.nn as nn
#import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ViewLayer(nn.Module):
    def __init__(self, channels, height, width):
        super(ViewLayer, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class cGAN(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size, class_num):
        super().__init__()

        self.z_size = z_size
        self.img_size = img_size

        # self.label_emb = nn.Embedding(class_num, class_num)

        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.img_size * self.img_size * 3),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)
        # One-hot vector to embedding vector
        # c = self.label_emb(labels)
        # Concat image & label
        c = labels
        print(f'c_shape_gen:{c.shape}')
        print(f'z_shape_gen:{z.shape}')
        x = torch.cat([z, c], 1)
        print(f'x_shape_gen:{x.shape}')
        # Generator out
        out = self.model(x)

        return out.view(-1, 3, self.img_size, self.img_size)


class cGANconv(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv, self).__init__()
        self.z_size = z_size
        self.features = 64
        self.img_size = img_size
        self.label_dim = 1
        self.img_channels = img_channels

        self.model = nn.Sequential(
            # Linear Layer as input
            nn.Linear(self.z_size + class_num, 128 * (self.img_size // 4) * self.img_size // 4),
            # Reshape to starting image dimensions (e.g., 128 x (img_size/4) x (img_size/4))
            ViewLayer(128, self.img_size // 4, self.img_size // 4),

            # Up-sampling layers
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer - adjust to match the number of output channels
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )


    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)

        c = labels
        x = torch.cat([z, c], 1)
        out = self.model(x)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)



class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_size, class_num):
        super().__init__()

        # self.label_emb = nn.Embedding(class_num, class_num)
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(3 * self.img_size * self.img_size + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Reshape fake image
        # print(f'x_shape: {x.shape}')
        x = x.view(-1, 3 * self.img_size * self.img_size)
        # print(x.shape)
        # One-hot vector to embedding vector
        # c = self.label_emb(labels)
        c = labels
        print(f'x_shape_disc:{x.shape}')
        print(f'c_shape_disc: {c.shape}')
        # Concat image & label
        x = torch.cat([x, c], 1)
        print(f'x_shape_disc: {x.shape}')
        # Discriminator out
        out = self.model(x)

        return out.squeeze()

