import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
from models.StackGAN import StackGANGen1, StackGANGen2, StackGANDisc1, StackGANDisc2

class StackGAN(pl.LightningModule):
    def __init__(self, condition_dim, num_classes, z_dim, gf_dim, df_dim):
        super().__init__()
        self.Gen1 = StackGANGen1(z_dim, condition_dim, gf_dim, num_classes)
        self.Gen2 = StackGANGen2(z_dim, condition_dim, gf_dim, num_classes)
        self.Disc1 = StackGANDisc1(condition_dim, df_dim)
        self.Disc2 = StackGANDisc2(condition_dim, df_dim)

    def forward(self, z, labels):
        # Stage 1 Generation
        low_res_imgs = self.Gen1(z, labels)

        # Stage 2 Generation
        high_res_imgs = self.Gen2(low_res_imgs, labels)

        return high_res_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, conditions = batch

        z = torch.randn(real_imgs.size(0))
        # Train Generators
        if optimizer_idx < 2:
            # Stage 1 Generator
            if optimizer_idx == 0:
                # Generate low-resolution images
                fake_imgs_stage1 = self.Gen1(z, conditions)
                # Pass to Discriminator 1
                pred_fake = self.Disc1(fake_imgs_stage1, conditions)
                # Generator 1 loss
                g_loss_1 = self.generator_loss(pred_fake)
                self.log('g_loss_1', g_loss_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return g_loss_1
            # Stage 2 Generator
            elif optimizer_idx == 1:
                # Generate high-resolution images
                fake_imgs_stage2 = self.Gen2(fake_imgs_stage1.detach(), conditions)
                # Pass to Discriminator 2
                pred_fake = self.Disc2(fake_imgs_stage2, conditions)
                # Generator 2 loss
                g_loss_2 = self.generator_loss(pred_fake)
                self.log('g_loss_2', g_loss_2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return g_loss_2
                # Train Discriminators
        else:
        # Discriminator 1
            if optimizer_idx == 2:
                # Real images loss
                pred_real = self.Disc1(real_imgs, conditions)
                d_loss_real = self.discriminator_loss(pred_real, True)

                # Fake images loss
                pred_fake = self.Disc1(fake_imgs_stage1.detach(), conditions)
                d_loss_fake = self.discriminator_loss(pred_fake, False)

                # Total loss
                d_loss_1 = (d_loss_real + d_loss_fake) / 2
                self.log('d_loss_1', d_loss_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return d_loss_1

                # Discriminator 2
            elif optimizer_idx == 3:
                # Real images loss
                pred_real = self.Disc2(real_imgs, conditions)
                d_loss_real = self.discriminator_loss(pred_real, True)

                # Fake images loss
                pred_fake = self.Disc2(fake_imgs_stage2.detach(), conditions)
                d_loss_fake = self.discriminator_loss(pred_fake, False)

                # Total loss
                d_loss_2 = (d_loss_real + d_loss_fake) / 2
                self.log('d_loss_2', d_loss_2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return d_loss_2

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # Create optimizers for both generators and discriminators
        optimizer_g1 = optim.Adam(self.Gen1.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_g2 = optim.Adam(self.Gen2.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d1 = optim.Adam(self.Disc1.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d2 = optim.Adam(self.Disc2.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Return optimizers and optionally, learning rate schedulers
        return [optimizer_g1, optimizer_g2, optimizer_d1, optimizer_d2], []
