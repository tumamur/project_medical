import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.cGAN import cGAN, cGANconv, EnhancedGenerator
from models.cGAN_discriminator import Discriminator
from models.Discriminator import ImageDiscriminator
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class CGANModule(pl.LightningModule):
    def __init__(self, num_classes, params):
        super(CGANModule, self).__init__()
        # Assuming cGANconv needs parameters to initialize
        #self.generator = cGANconv(z_size=params["image_generator"]["z_size"], img_size=params["dataset"]["input_size"],
         #            class_num=params["trainer"]["num_classes"],
          #           img_channels=params["image_discriminator"]["channels"])
        self.generator = EnhancedGenerator(z_size=params["image_generator"]["z_size"],
                                           img_size=params["dataset"]["input_size"],
                                           img_channels=params["image_discriminator"]["channels"],
                                           class_num=params["trainer"]["num_classes"])
        size = params["dataset"]["input_size"]
        # self.discriminator = ImageDiscriminator((3, size, size))
        self.discriminator = Discriminator(img_size=params["dataset"]["input_size"],
                                           class_num=params["trainer"]["num_classes"],
                                           img_channels=params["image_discriminator"]["channels"])
        self.class_num = num_classes
        self.batch_size = params["dataset"]["batch_size"]
        self.z_dim = params["image_generator"]["z_size"]
        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCELoss()

    def forward(self):
        pass

    def discriminator_step(self, batch, batch_idx, real_validity, fake_validity):
        real_loss = self.criterion(real_validity, Variable(torch.ones(real_validity.size())).to(self.device))
        self.log("real_disc_loss", real_loss, on_step=True, on_epoch=True, prog_bar=True)
        fake_loss = self.criterion(fake_validity, Variable(torch.zeros(fake_validity.size())).to(self.device))
        self.log("fake_disc_loss", fake_loss, on_step=True, on_epoch=True, prog_bar=True)
        loss = (real_loss + fake_loss) / 2
        return loss

    def generator_step(self, batch, batch_idx, validity):
        loss = self.criterion(validity, Variable(torch.ones(validity.size())).to(self.device))
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images = batch['target'].float()
        labels = batch['report'].float()
        batch_nmb = real_images.shape[0]
        z = Variable(torch.randn(batch_nmb, self.z_dim).to(self.device))
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, self.class_num, self.batch_size))).to(self.device)
        # Generate fake images
        fake_images = self.generator(z, labels)

        # Discriminator predictions
        # fake_validity = self.discriminator(fake_images, labels)
        fake_validity = self.discriminator(fake_images, labels)
        # real_validity = self.discriminator(real_images, labels)
        real_validity = self.discriminator(real_images, labels)

        if batch_idx % 30 == 0:
            grid = torchvision.utils.make_grid(fake_images[:4]).detach().cpu()  # Show first 4 images
            grid = grid.permute(1, 2, 0)  # Rearrange channels for plotting
            grid = grid * 0.5 + 0.5  # Undo normalization if applied
            plt.imshow(grid.numpy())
            plt.show()

        if optimizer_idx == 0:
            loss = self.generator_step(batch, batch_idx, fake_validity)
            self.log('gen_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        elif optimizer_idx == 1 and batch_idx % 5 == 0:
            loss = self.discriminator_step(batch, batch_idx, real_validity, fake_validity)
            self.log('disc_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        real_images = batch['target'].float()
        labels = batch['report'].float()
        batch_nmb = real_images.shape[0]
        z = Variable(torch.randn(batch_nmb, self.z_dim).to(self.device))
        # fake_labels = Variable(torch.LongTensor(np.random.randint(0, self.class_num, self.batch_size))).to(self.device)
        # Generate fake images
        fake_images = self.generator(z, labels)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(fake_images[:4]).detach().cpu()  # Show first 4 images
            grid = grid.permute(1, 2, 0)  # Rearrange channels for plotting
            grid = grid * 0.5 + 0.5  # Undo normalization if applied
            plt.imshow(grid.numpy())
            plt.show()

        # Discriminator predictions for fake images
        fake_validity = self.discriminator(fake_images, labels)
        fake_loss = self.criterion(fake_validity, Variable(torch.zeros(fake_validity.size())).to(self.device))

        # Discriminator predictions for real images
        real_validity = self.discriminator(real_images, labels)
        real_loss = self.criterion(real_validity, Variable(torch.ones(real_validity.size())).to(self.device))

        # Average loss
        loss = (fake_loss + real_loss) / 2

        # Logging the validation loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # Implement test step if needed
        pass

    def configure_optimizers(self):
        # You might want separate optimizers for generator and discriminator
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        return [gen_optimizer, disc_optimizer], []
