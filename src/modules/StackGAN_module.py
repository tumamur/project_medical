import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from models.StackGAN import StackGANGen1, StackGANGen2, StackGANDisc1, StackGANDisc2
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class StackGAN(pl.LightningModule):
    def __init__(self, condition_dim, num_classes, z_dim, gf_dim, df_dim, r_num):
        super().__init__()
        self.Gen1 = StackGANGen1(z_dim, condition_dim, gf_dim, num_classes)
        self.Gen2 = StackGANGen2(z_dim, condition_dim, gf_dim, num_classes, r_num)
        self.Disc1 = StackGANDisc1(condition_dim, df_dim)
        self.Disc2 = StackGANDisc2(condition_dim, df_dim)
        self.criterion = nn.MSELoss()
        self.z_dim = z_dim
        self.condition_dim = condition_dim
        self.num_classes = num_classes
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.r_num = r_num

        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
        ])

    def forward(self, z, labels):
        # Stage 1 Generation
        low_res_imgs = self.Gen1(z, labels)

        # Stage 2 Generation
        high_res_imgs = self.Gen2(low_res_imgs, labels)

        return high_res_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_img = batch['target'].float()
        real_img_low = self.transforms(real_img)
        real_report = batch['report'].float()
        batch_nmb = real_img.shape[0]
        # z = Variable(torch.FloatTensor(batch_nmb, self.z_dim)).to(self.device)
        z = torch.randn(batch_nmb, self.z_dim).to(self.device)
        # z = torch.full((batch_nmb, self.z_dim), fill_value=0.1).to(self.device)  # Controlled small values
        # print(z)

        valid = torch.ones(batch_nmb, device=self.device)
        fake = torch.zeros(batch_nmb, device=self.device)

        # Train Generators
        if optimizer_idx < 2:
            # Stage 1 Generator
            if optimizer_idx == 0 and self.current_epoch < 100:
                # _, fake_img_stage1, mu, logvar = self.Gen1(z, real_report)
                _, fake_img_stage1, c_code = self.Gen1(z, real_report)
                # Pass to Discriminator 1
                # pred_fake = self.Disc1(fake_img_stage1, real_report)
                pred_fake_features = self.Disc1(fake_img_stage1)
                # pred_fake = self.Disc1.get_cond_logits(pred_fake_features, mu)
                # pred_fake = self.Disc1.get_cond_logits(pred_fake_features, c_code)
                pred_fake = self.Disc1.get_uncond_logits(pred_fake_features)
                if batch_idx == 0:
                    self.visualize(fake_img_stage1, real_img_low)
                # Generator 1 loss
                g_loss_1 = self.criterion(pred_fake, valid)
                self.log('g_loss_1', g_loss_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                if g_loss_1 > 0.1:
                    return g_loss_1

            # Stage 2 Generator
            elif optimizer_idx == 1 and self.current_epoch >= 100:
                # Generate high-resolution images
                # _, fake_img_stage2, mu, logvar = self.Gen2(z, real_report)
                _, fake_img_stage2, c_code = self.Gen2(z, real_report)
                # Pass to Discriminator 2
                # pred_fake = self.Disc2(fake_img_stage2, real_report)
                pred_fake_features = self.Disc2(fake_img_stage2)
                # pred_fake = self.Disc2.get_cond_logits(pred_fake_features, mu)
                # pred_fake = self.Disc2.get_cond_logits(pred_fake_features)
                pred_fake = self.Disc2.get_uncond_logits(pred_fake_features)
                # Generator 2 loss
                # print(f'pred_fake stage 2: {pred_fake}')
                g_loss_2 = self.criterion(pred_fake, valid)
                self.log('g_loss_2', g_loss_2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                if g_loss_2 > 0.1:
                    return g_loss_2

                # Train Discriminators
        else:
        # Discriminator 1
            if optimizer_idx == 2 and self.current_epoch < 100:
                _, fake_img_stage1, c_code = self.Gen1(z, real_report)
                # Real images loss
                # pred_real = self.Disc1(real_img_low, real_report)
                pred_real_features = self.Disc1(real_img_low)
                # pred_real = self.Disc1.get_cond_logits(pred_real_features, mu)
                # pred_real = self.Disc1.get_cond_logits(pred_real_features, c_code)
                # pred_real = self.Disc1.get_cond_logits(pred_real_features)
                pred_real = self.Disc1.get_uncond_logits(pred_real_features)
                d_loss_real = self.criterion(pred_real, valid)

                # Fake images loss
                # pred_fake = self.Disc1(fake_img_stage1.detach(), real_report)
                pred_fake_features = self.Disc1(fake_img_stage1)
                # pred_fake = self.Disc1.get_cond_logits(pred_fake_features, mu)
                # pred_fake = self.Disc1.get_cond_logits(pred_fake_features, c_code)
                # pred_fake = self.Disc1.get_cond_logits(pred_fake_features)
                pred_fake = self.Disc1.get_uncond_logits(pred_fake_features)
                d_loss_fake = self.criterion(pred_fake, fake)

                # Total loss
                d_loss_1 = (d_loss_real + d_loss_fake) / 2
                self.log('d_loss_1', d_loss_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                if d_loss_1 > 0.1:
                    return d_loss_1

            # Discriminator 2
            elif optimizer_idx == 3 and self.current_epoch >= 100:
                # Real images loss
                # Generate high-resolution images
                # img_stage_1, fake_img_stage2, mu, logvar = self.Gen2(z, real_report)
                img_stage1, fake_img_stage2, c_code = self.Gen2(z, real_report)
                # pred_real = self.Disc2(real_img, real_report)
                pred_real_features = self.Disc2(real_img)
                # pred_real = self.Disc2.get_cond_logits(pred_real_features, mu)
                # pred_real = self.Disc2.get_cond_logits(pred_real_features, c_code)
                pred_real = self.Disc2.get_uncond_logits(pred_real_features)
                d_loss_real = self.criterion(pred_real, valid)

                # Fake images loss
                # pred_fake = self.Disc2(fake_img_stage2.detach(), real_report)
                pred_fake_features = self.Disc2(fake_img_stage2)
                # pred_fake = self.Disc2.get_cond_logits(pred_fake_features, mu)
                # pred_fake = self.Disc1.get_cond_logits(pred_fake_features, c_code)
                pred_fake = self.Disc2.get_uncond_logits(pred_fake_features)
                d_loss_fake = self.criterion(pred_fake, fake)

                # Total loss
                d_loss_2 = (d_loss_real + d_loss_fake) / 2
                self.log('d_loss_2', d_loss_2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                if d_loss_2 > 0.1:
                    return d_loss_2

    def validation_step(self, batch, batch_idx):
        pass

    def visualize(self, fake_image, real_image):
        fake_image = self.convert_tensor_to_image(fake_image[0])
        plt.imshow(fake_image)
        plt.axis('off')
        plt.show()

        real_image = self.convert_tensor_to_image(real_image[0])
        plt.imshow(real_image)
        plt.axis('off')
        plt.show()

    def convert_tensor_to_image(self, tensor):
        # Denormalize and convert to PIL Image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denorm = tensor.clone().cpu().detach()
        for t, m, s in zip(denorm, mean, std):
            t.mul_(s).add_(m)
        denorm = denorm.numpy().transpose(1, 2, 0)
        denorm = np.clip(denorm, 0, 1)
        return Image.fromarray((denorm * 255).astype('uint8'))

    def configure_optimizers(self):
        # Create optimizers for both generators and discriminators
        optimizer_g1 = optim.Adam(self.Gen1.parameters(), lr=0.00002, betas=(0.9, 0.999))
        optimizer_g2 = optim.Adam(self.Gen2.parameters(), lr=0.00002, betas=(0.9, 0.999))
        optimizer_d1 = optim.Adam(self.Disc1.parameters(), lr=0.00002, betas=(0.9, 0.999))
        optimizer_d2 = optim.Adam(self.Disc2.parameters(), lr=0.00002, betas=(0.9, 0.999))

        # Return optimizers and optionally, learning rate schedulers
        return [optimizer_g1, optimizer_g2, optimizer_d1, optimizer_d2], []
