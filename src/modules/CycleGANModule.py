import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
import numpy as np
import deepspeed as ds
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from PIL import Image

from models.BioViL import BioViL
from models.StackGAN import StackGANGen1, StackGANGen2, StackGANDisc1, StackGANDisc2
from models.ARK import ARKModel
from utils.buffer import ReportBuffer, ImageBuffer
from models.Discriminator import ImageDiscriminator, ReportDiscriminator
from models.cGAN import cGAN, cGANconv
from models.DDPM import ContextUnet, DDPM
from utils.environment_settings import env_settings
from losses.Losses import *
from losses.Metrics import *


import os
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
torch.autograd.set_detect_anomaly(True)



class CycleGAN(pl.LightningModule):
    """
    CycleGAN class for image-to-report translation and vice versa.
    This class contains both generators and discriminators for image and report generation.
    """

    def __init__(self, opt, val_dataloader):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.save_hyperparameters(opt)
        self.initialize_components()
        self.define_loss_functions()
        self.define_networks()
        self.define_optimizers()

    def initialize_components(self):
        """
        Initialize generators, discriminators, and buffers.
        """
        # Extract options
        self.lambda_I = 10.0
        self.lambda_R = 10.0
        self.discriminator_update_freq = 30
        self.data_imputation = self.opt['dataset']['data_imputation']
        self.input_size = self.opt["dataset"]["input_size"]
        self.num_classes = self.opt['trainer']['num_classes']
        self.buffer_size = self.opt['trainer']["buffer_size"]
        self.z_size = self.opt["image_generator"]["z_size"]
        self.batch_size = self.opt["dataset"]["batch_size"]
        self.log_images_steps = self.opt["trainer"]["log_images_steps"]
        self.n_epochs = self.opt['trainer']['n_epoch']
        self.n_feat = self.opt["image_generator"]["n_feat"]
        self.n_T = self.opt["image_generator"]["n_T"]

    def define_networks(self):
        self.report_generator = self._get_report_generator()
        self.image_generator = self._get_image_generator()
        self.report_discriminator = self._get_report_discriminator()
        self.image_discriminator = self._get_image_discriminator()
        
        self.buffer_reports = ReportBuffer(self.buffer_size)
        self.buffer_images = ImageBuffer(self.buffer_size)


    def define_optimizers(self):

        # Initialize optimizer dictionary
        self.optimizer_dict = {
            'Adam': torch.optim.Adam, 
            'AdamW': torch.optim.AdamW,
        }
        
        # Optimizers
        self.image_gen_optimizer = self.optimizer_dict[self.opt["image_generator"]["optimizer"]]
        self.report_gen_optimizer = self.optimizer_dict[self.opt["report_generator"]["optimizer"]]
        self.image_disc_optimizer = self.optimizer_dict[self.opt["image_discriminator"]["optimizer"]]
        self.report_disc_optimizer = self.optimizer_dict[self.opt["report_discriminator"]["optimizer"]]


    def define_loss_functions(self):
        """
        Define and initialize loss functions.
        """
        # Initialize loss functions

        self.loss_dict = { 
            #'ark' : Loss('ark', ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK']),
            #'biovil' : Loss('biovil'),
            'biovil_t' : Loss('biovil_t'),
            #'vgg' : Loss('vgg'),
            #'style_vgg' : Loss('style_vgg'),
            #'cosine_similarity' : Loss('classification'),
            'MSE' : Loss('MSE'),
            'BCE' : Loss('BCE'),
            'L1' : Loss('L1')
        }

        self.img_consistency_loss = self.loss_dict[self.opt['image_generator']['consistency_loss']]
        self.img_adversarial_loss = self.loss_dict[self.opt['image_discriminator']['adversarial_loss']]
        self.report_consistency_loss = self.loss_dict[self.opt['report_generator']['consistency_loss']]
        self.report_adversarial_loss = self.loss_dict[self.opt['report_discriminator']['adversarial_loss']]


    def forward(self, img):
        img = img.float().to(self.device)
        generated_report = self.report_generator(img).float()
        #z = Variable(torch.randn(self.batch_size, self.z_size)).float().to(self.device)
        #generated_img = self.image_generator(z, report)
        #return generated_report, generated_img
        return None
        
    def configure_optimizers(self):
        
        optimizer_img_gen = self.image_gen_optimizer(self.image_generator.parameters(),
                                                     lr = self.opt['image_generator']['learning_rate'], 
                                                     betas=(self.opt['image_generator']['beta'], 0.999))

        optimizer_report_gen = self.report_gen_optimizer(self.report_generator.parameters(), 
                                                        lr = self.opt['report_generator']['learning_rate'], 
                                                        betas=(self.opt['report_generator']['beta'], 0.999))
                                                        
        optimizer_img_disc = self.image_disc_optimizer(self.image_discriminator.parameters(), 
                                                        lr = self.opt['image_discriminator']['learning_rate'], 
                                                        betas=(self.opt['image_discriminator']['beta'], 0.999))
                                                       
        optimizer_report_disc = self.report_disc_optimizer(self.report_discriminator.parameters(), 
                                                        lr = self.opt['report_discriminator']['learning_rate'], 
                                                        betas=(self.opt['report_discriminator']['beta'], 0.999))

        return [optimizer_img_gen, optimizer_report_gen, optimizer_img_disc, optimizer_report_disc]



    def log_gen_metrics(self,loss):
        # TODO : Log only avg values
        total_loss, loss_IR, loss_RI, loss_IRI, loss_RIR = loss
        metrics = {
            'loss_G' : total_loss,
            'loss_IR' : loss_IR,
            'loss_RI' : loss_RI,
            'loss_IRI' : loss_IRI,
            'loss_RIR' : loss_RIR
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

                                                           
    
    def generator_step(self, valid_img, valid_report):
        # adversarial_losses = GAN Loss
        loss_IR = self.report_adversarial_loss(self.report_discriminator(self.fake_report), valid_report)
        loss_RI = self.img_adversarial_loss(self.image_discriminator(self.fake_img), valid_img)
        # consistency_losses = Cycle Loss
        loss_IRI_perceptual = self.img_consistency_loss(self.real_img, self.cycle_img)
        loss_IRI_L1 = self.loss_dict['L1'](self.real_img, self.cycle_img)
        loss_IRI = (loss_IRI_perceptual + loss_IRI_L1) * self.lambda_I
        
        loss_RIR = self.report_consistency_loss(self.cycle_report, self.real_report) * self.lambda_R
        loss = loss_IR + loss_RI + loss_IRI + loss_RIR
        self.log_gen_metrics((loss, loss_IR, loss_RI, loss_IRI, loss_RIR))
        
        return loss
        

    def report_discriminator_step(self, valid, fake):
        fake_report = self.buffer_reports(self.fake_report)
        # discriminator_loss
        loss_real = self.report_adversarial_loss(self.report_discriminator(self.real_report), valid)
        loss_fake = self.report_adversarial_loss(self.report_discriminator(self.fake_report.detach()), fake)
        loss = (loss_real + loss_fake) * 0.5
        self.log('loss_D_R', loss, on_step=True)
        return loss

    def img_discriminator_step(self, valid, fake):
        fake_image = self.buffer_images(self.fake_img)
        #discriminator loss
        loss_real = self.img_adversarial_loss(self.image_discriminator(self.real_img), valid)
        loss_fake = self.img_adversarial_loss(self.image_discriminator(self.fake_img), fake)
        loss = (loss_real + loss_fake) * 0.5
        self.log('loss_D_I', loss, on_step=True)
        return loss
        

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)


        # DDPM 
        if self.opt['image_generator']['model'] == 'ddpm':
            _ts = torch.randint(1, self.n_T+1, (self.real_img.shape[0], )).to(self.device)
            noise = torch.randn_like(self.real_img).to(self.device)
            z = (_ts, noise, self.real_img)

        
        # generate valid and fake labels
        valid_img_sample = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake_img_sample = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))

        
        valid_report_sample = Tensor(np.ones((self.real_report.size(0), *self.report_discriminator.output_shape)))
        fake_report_sample = Tensor(np.zeros((self.real_report.size(0), *self.report_discriminator.output_shape)))

        # get generated image and reports from the generators
        self.fake_report = self.report_generator(self.real_img)
        self.fake_img = self.image_generator(z, self.real_report)
  
        fake_reports = torch.sigmoid(self.fake_report)
        # TODO : Experiment without torch.where
        fake_reports = torch.where(fake_reports < 0.5, torch.tensor(0.0, device=fake_reports.device), fake_reports)
        fake_reports = torch.where(fake_reports >= 0.5, torch.tensor(1.0, device=fake_reports.device), fake_reports)

        # reconstruct reports and images

        if self.opt['image_generator']['model'] == 'ddpm':
            _ts = torch.randint(1, self.n_T+1, (self.fake_img.shape[0], )).to(self.device)
            noise = torch.randn_like(self.fake_img).to(self.device)
            z = (_ts, noise, self.fake_img)

        
        self.fake_img = ddpm.sample(self.noise)
        self.cycle_report = self.report_generator(self.fake_img)

        
        self.cycle_img = self.image_generator(z, fake_reports)
        

        if (batch_idx % self.log_images_steps) == 0 and optimizer_idx == 0:
            self.log_images_on_cycle(batch_idx)
            self.log_reports_on_cycle(batch_idx)
            self.visualize_images(batch_idx)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_loss = self.generator_step(valid_img=valid_img_sample, valid_report=valid_report_sample)
            return {"loss": gen_loss}
        
        update_discriminator = (batch_idx % self.discriminator_update_freq) == 0
        if (optimizer_idx == 2 or optimizer_idx == 3) and update_discriminator:
            img_disc_loss = self.img_discriminator_step(valid_img_sample, fake_img_sample)
            report_disc_loss = self.report_discriminator_step(valid_report_sample, fake_report_sample)
            disc_loss = img_disc_loss + report_disc_loss
            return {"loss": disc_loss}

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        # return generated report and generated image from generated report
        report, image = self(batch['target'])
        return report, image
    
    def log_images_on_cycle(self, batch_idx):

        cycle_img_1 = self.cycle_img[0]
        real_img_1 = self.real_img[0]
        fake_img_1 = self.fake_img[0]

        cycle_img_tensor = cycle_img_1
        real_img_tensor = real_img_1
        fake_img_tensor = fake_img_1

        step = self.current_epoch * batch_idx + batch_idx

        self.logger.experiment.add_image(f"On step cycle img", cycle_img_tensor, step, dataformats='CHW')
        self.logger.experiment.add_image(f"On step real img", real_img_tensor, step, dataformats='CHW')
        self.logger.experiment.add_image(f"On step fake_img", fake_img_tensor, step, dataformats='CHW')

    def log_reports_on_cycle(self, batch_idx):
        real_report = self.real_report[0]
        cycle_report = self.cycle_report[0]
        # Process the generated report
        real_report = real_report.cpu().detach()
        real_report = torch.sigmoid(real_report)
        real_report = (real_report > 0.5).int()
        report_text_labels = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(real_report) if
                              val == 1]
        report_text_real = ', '.join(report_text_labels)

        generated_report = cycle_report.cpu().detach()
        generated_report = torch.sigmoid(generated_report)
        generated_report = (generated_report > 0.5).int()
        report_text_labels_cycle = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(generated_report) if
                              val == 1]
        report_text_cycle = ', '.join(report_text_labels_cycle)

        step = self.current_epoch * batch_idx + batch_idx

        self.logger.experiment.add_text(f"On step cycle report", report_text_real, step)
        self.logger.experiment.add_text(f"On step real report", report_text_cycle, step)

    def visualize_images(self, batch_idx):
        tensor = self.real_img[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denorm = tensor.clone().cpu().detach()
        for t, m, s in zip(denorm, mean, std):
            t.mul_(s).add_(m)

        image_to_display = denorm.numpy().transpose(1, 2, 0)
        image_to_display = np.clip(image_to_display, 0, 1)
        # plt.imshow(tensor.permute(1, 2, 0).cpu().detach())
        plt.imshow(image_to_display)
        plt.axis('off')
        plt.show()

        cycle_tensor = self.cycle_img[0]
        denorm = cycle_tensor.clone().cpu().detach()
        for t, m, s in zip(denorm, mean, std):
            t.mul_(s).add_(m)
        image_to_display = denorm.numpy().transpose(1, 2, 0)
        image_to_display = np.clip(image_to_display, 0, 1)
        # plt.imshow(cycle_tensor.permute(1, 2, 0).cpu().detach())
        plt.imshow(image_to_display)
        plt.axis('off')
        plt.show()

    def _get_report_generator(self):
        model_dict = {
            # 'ark' : ARKModel(num_classes=self.num_classes,
            #                    ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK']),
            'biovil_t': BioViL(embedding_size=self.opt["report_generator"]["embedding_size"], 
                              num_classes=self.num_classes, 
                              hidden_1=self.opt["report_generator"]["classification_head_hidden1"],
                              hidden_2=self.opt["report_generator"]["classification_head_hidden2"], 
                              dropout_rate=self.opt["report_generator"]["dropout_prob"])
        }

        return model_dict[self.opt['report_generator']['model']]

    def _get_report_discriminator(self):
        return ReportDiscriminator(input_dim=self.num_classes)
    
    def _get_image_generator(self):

        #C.GAN.EMBEDDING_DIM = 128
        # __C.GAN.DF_DIM = 64
        # __C.GAN.GF_DIM = 128

        model_dict = {
            'cgan' : cGANconv(z_size=self.z_size, img_size=self.input_size, class_num=self.num_classes,
                    img_channels=self.opt["image_discriminator"]["channels"]) ,
            'ddpm' : DDPM(nn_model=ContextUnet(in_channels=3, n_feat=self.n_feat, n_classes=self.num_classes), 
                          image_size=(self.input_size, self.input_size),
                          betas=(float(self.opt['image_generator']['ddpm_beta1']), 
                                 float(self.opt['image_generator']['ddpm_beta2'])),
                          n_T=self.n_T,
                          drop_prob=self.opt['image_generator']['ddpm_drop_prob']),
            
            'stackgan_gen1' : None
        }
        return model_dict[self.opt["image_generator"]["model"]]
        
    def _get_image_discriminator(self):
        return ImageDiscriminator(input_shape=(self.opt['image_discriminator']['channels'], 
                                               self.opt['image_discriminator']['img_height'],
                                               self.opt['image_discriminator']['img_width'])
                                 )