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
from cGAN import Discriminator, Generator


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('torch version:', torch.__version__)
    print('device:', device)

    img_size = 224  # Image size
    batch_size = 64  # Batch size

    # Model
    z_size = 100
    generator_layer_size = [256, 512, 1024]
    discriminator_layer_size = [1024, 512, 256]

    # Training
    epochs = 30  # Train epochs
    learning_rate = 1e-4

    class_num = 13

    # Define generator
    generator = Generator(generator_layer_size, z_size, img_size, class_num).to(device)
    # Define discriminator
    discriminator = Discriminator(discriminator_layer_size, img_size, class_num).to(device)

    # Optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
        # Init gradient
        g_optimizer.zero_grad()

        # Building z
        z = Variable(torch.randn(batch_size, z_size)).to(device)

        # Building fake labels
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)

        # Generating fake images
        fake_images = generator(z, fake_labels)

        # Disciminating fake images
        validity = discriminator(fake_images, fake_labels)

        # Calculating discrimination loss (fake images)
        g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))

        # Backword propagation
        g_loss.backward()

        #  Optimizing generator
        g_optimizer.step()

        return g_loss.data

    def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
        # Init gradient
        d_optimizer.zero_grad()

        # Disciminating real images
        real_validity = discriminator(real_images, labels)

        # Calculating discrimination loss (real images)
        real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

        # Building z
        z = Variable(torch.randn(batch_size, z_size)).to(device)

        # Building fake labels
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)

        # Generating fake images
        fake_images = generator(z, fake_labels)

        # Disciminating fake images
        fake_validity = discriminator(fake_images, fake_labels)

        # Calculating discrimination loss (fake images)
        fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

        # Sum two losses
        d_loss = real_loss + fake_loss

        # Backword propagation
        d_loss.backward()

        # Optimizing discriminator
        d_optimizer.step()

        return d_loss.data

    for epoch in range(epochs):

        print('Starting epoch {}...'.format(epoch + 1))

        for i, (images, labels) in enumerate(data_loader):
            # Train data
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Set generator train
            generator.train()

            # Train discriminator
            d_loss = discriminator_train_step(len(real_images), discriminator,
                                              generator, d_optimizer, criterion,
                                              real_images, labels)

            # Train generator
            g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

        # Set generator eval
        generator.eval()

        print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))

        # Building z
        z = Variable(torch.randn(class_num - 1, z_size)).to(device)

        # Labels 0 ~ 8
        labels = Variable(torch.LongTensor(np.arange(class_num - 1))).to(device)

        # Generating images
        sample_images = generator(z, labels).unsqueeze(1).data.cpu()

        # Show images
        grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
        plt.imshow(grid)
        plt.show()

