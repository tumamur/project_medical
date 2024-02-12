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

from models.BioViL import *
from models.StackGAN import StackGANGen1, StackGANGen2, StackGANDisc1, StackGANDisc2
from models.ARK import ARKModel
from utils.buffer import ReportBuffer, ImageBuffer, convert_to_soft_labels
from models.Discriminator import *
from models.cGAN import cGAN, cGANconv
from models.DDPM import ContextUnet, DDPM
from utils.environment_settings import env_settings
from losses.Losses import *
from losses.Metrics import *
from torchmetrics import Accuracy, Precision, Recall, F1Score


import os
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
torch.autograd.set_detect_anomaly(True)



class CycleGAN(pl.LightningModule):
    """
    CycleGAN class for image-to-report translation and vice versa.
    This class contains both generators and discriminators for image and report generation.

    Generator for image-to-report translation (report_generator).
    Generator for report-to-image translation (image_generator).
    Discriminator for Generated Reports (report_dicriminator). 
    Discriminator for Generated Images (image_generator).
        
    """

    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.save_hyperparameters(opt)
        self.initialize_components()
        self.define_loss_functions()
        self.define_networks()
        self.define_metrics()
        self.define_optimizers()

    def initialize_components(self):
        """
        Initialize generators, discriminators, and buffers.
        """
        # Extract options
        self.lambda_I = 10.0
        self.lambda_R = 10.0
        self.update_freq = self.opt['trainer']['update_freq']
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
        self.soft_label_type = self.opt['trainer']['soft_label_type']   

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

    def define_metrics(self):
        self.val_metrics = {
            'accuracy_micro' : Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'precision_micro' : Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'recall_micro' : Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'f1_micro' : F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'accuracy_macro' : Accuracy(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'precision_macro' : Precision(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'recall_macro' : Recall(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'f1_macro' : F1Score(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'overall_precision' : [],
        }


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
        return generated_report
        
    def configure_lr_schedulers(self, optimizer):
        lr_scheduler_dict = {
            "cosine_lr": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-7),
            "step_lr": torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                       step_size=self.opt['trainer']['scheduler_iter'],
                                                       gamma=self.opt['trainer']['step_lr_gamma']),
            "exp_lr": torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                             gamma=self.opt['trainer']['exp_lr_gamma']),
            'lambda': None,
            'ReduceLROnPlateau': None
        }
        return lr_scheduler_dict[self.opt['trainer']['lr_scheduler']]

    def configure_optimizers(self):

        if self.opt['report_generator']['decay']:
            optimizer_report_gen = self.report_gen_optimizer(self.report_generator.parameters(),
                                                             lr=self.opt['report_generator']['learning_rate'],
                                                             betas=(self.opt['report_generator']['beta'], 0.999),
                                                             weight_decay=self.opt['report_generator']['weight_decay'])

        else:
            optimizer_report_gen = self.report_gen_optimizer(self.report_generator.parameters(),
                                                             lr=self.opt['report_generator']['learning_rate'],
                                                             betas=(self.opt['report_generator']['beta'], 0.999))

        optimizer_img_gen = self.image_gen_optimizer(self.image_generator.parameters(),
                                                     lr=self.opt['image_generator']['learning_rate'],
                                                     betas=(self.opt['image_generator']['beta'], 0.999))

        optimizer_img_disc = self.image_disc_optimizer(self.image_discriminator.parameters(),
                                                       lr=self.opt['image_discriminator']['learning_rate'],
                                                       betas=(self.opt['image_discriminator']['beta'], 0.999))

        optimizer_report_disc = self.report_disc_optimizer(self.report_discriminator.parameters(),
                                                           lr=self.opt['report_discriminator']['learning_rate'],
                                                           betas=(self.opt['report_discriminator']['beta'], 0.999))

        # Initialize schedulers
        scheduler_img_gen = self.configure_lr_schedulers(optimizer_img_gen)
        scheduler_report_gen = self.configure_lr_schedulers(optimizer_report_gen)
        scheduler_img_disc = self.configure_lr_schedulers(optimizer_img_disc)
        scheduler_report_disc = self.configure_lr_schedulers(optimizer_report_disc)

        optimizers = [optimizer_img_gen, optimizer_report_gen, optimizer_img_disc, optimizer_report_disc]
        lr_schedulers = [scheduler_img_gen, scheduler_report_gen, scheduler_img_disc, scheduler_report_disc]

        return optimizers, lr_schedulers


    def log_gen_loss(self,loss):
        # TODO : Log only avg values
        total_loss, loss_IR, loss_RI, loss_IRI, loss_RIR = loss
        metrics = {
            'loss_G_total' : total_loss,
            'loss_IR' : loss_IR,
            'loss_RI' : loss_RI,
            'loss_IRI' : loss_IRI,
            'loss_RIR' : loss_RIR
        }
        # update key names with phase

        metrics = {f'{self.phase}_{key}': value for key, value in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

    def log_val_metrics(self, metrics, on_step):
        # log the metrics
        self.log_dict(metrics, on_step=on_step, on_epoch=True, prog_bar=True)

                                                           
    
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
        self.log_gen_loss((loss, loss_IR, loss_RI, loss_IRI, loss_RIR))
        
        return loss
        

    def report_discriminator_step(self, valid, fake):
        fake_report = self.buffer_reports(self.fake_report)
        # discriminator_loss
        loss_real = self.report_adversarial_loss(self.report_discriminator(self.real_report), valid)
        loss_fake = self.report_adversarial_loss(self.report_discriminator(fake_report.detach()), fake)
        loss = (loss_real + loss_fake) * 0.5
        self.log(self.phase + '_loss_D_R', loss, on_step=True)
        return loss

    def img_discriminator_step(self, valid, fake):
        fake_image = self.buffer_images(self.fake_img)
        #discriminator loss
        loss_real = self.img_adversarial_loss(self.image_discriminator(self.real_img), valid)
        loss_fake = self.img_adversarial_loss(self.image_discriminator(fake_image), fake)
        loss = (loss_real + loss_fake) * 0.5
        self.log(self.phase + '_loss_D_I', loss, on_step=True)
        return loss
        

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.phase = 'train'
        self.real_img = batch['target'].float()
        hard_report = batch['report'].float()
        if self.soft_label_type is not None:
            soft_report = convert_to_soft_labels(self.soft_label_type, hard_report, self.current_epoch)
            self.real_report = soft_report
        else:
            self.real_report = hard_report
        
        batch_nmb = self.real_img.shape[0]

        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)
        
        # generate valid and fake labels
        valid_img_sample = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake_img_sample = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))

        
        valid_report_sample = Tensor(np.ones((self.real_report.size(0), *self.report_discriminator.output_shape)))
        fake_report_sample = Tensor(np.zeros((self.real_report.size(0), *self.report_discriminator.output_shape)))

        # get generated image and reports from the generators
        self.fake_report = self.report_generator(self.real_img)

        
        self.fake_img = self.image_generator(z, self.real_report)
  
        fake_reports = torch.sigmoid(self.fake_report)
        if not self.opt['trainer']['use_float_reports']:
            fake_reports = torch.where(fake_reports > 0.5, 1, 0)

        self.fake_img = self.image_generator(z, self.real_report)
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator(z, fake_reports)


        if ((batch_idx+1)*self.batch_size) % self.update_freq == 0:
            # plot the images and reports
            self.log_images_on_cycle(batch_idx)
            self.log_reports_on_cycle(batch_idx)
            self.visualize_images(batch_idx)
            if self.opt['trainer']['save_images']:
                self.save_images(batch_idx)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_loss = self.generator_step(valid_img=valid_img_sample, valid_report=valid_report_sample)
            if gen_loss > self.opt['trainer']['gen_threshold']:
                return {"loss": gen_loss}

        elif optimizer_idx == 2:
            img_disc_loss = self.img_discriminator_step(valid_img_sample, fake_img_sample)
            if img_disc_loss > self.opt['trainer']['img_disc_threshold']:
                return {"loss": img_disc_loss}

        elif optimizer_idx == 3:
            report_disc_loss = self.report_discriminator_step(valid_report_sample, fake_report_sample)
            if report_disc_loss > self.opt['trainer']['report_disc_threshold']:
                return {"loss": report_disc_loss}

    def calculate_overall_precision(self, preds, targets, batch_nmb):
        exact_matches = torch.all(preds == targets, dim=1)
        true_positives = torch.sum(exact_matches).item()
        precision = true_positives / batch_nmb
        return precision

    def validation_step(self, batch, batch_idx):

        self.phase = 'val'

        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        self.fake_report = self.report_generator(self.real_img)
        self.fake_report = torch.sigmoid(self.fake_report)
        #self.fake_report_0_1 = torch.where(self.fake_report > 0.5, 1, 0)
        self.fake_report_0_1 = torch.where(self.fake_report < 0.5, torch.tensor(0.0, device=self.fake_report.device), self.fake_report)
        self.fake_report_0_1 = torch.where(self.fake_report_0_1 >= 0.5, torch.tensor(1.0, device=self.fake_report.device), self.fake_report_0_1)
        
        # update the metrics
        self.val_metrics['accuracy_micro'].update(self.fake_report_0_1, self.real_report)
        self.val_metrics['precision_micro'].update(self.fake_report_0_1, self.real_report)
        self.val_metrics['recall_micro'].update(self.fake_report_0_1, self.real_report)
        self.val_metrics['f1_micro'].update(self.fake_report_0_1, self.real_report)

        self.val_metrics['accuracy_macro'].update(self.fake_report_0_1, self.real_report)
        self.val_metrics['precision_macro'].update(self.fake_report_0_1, self.real_report)
        self.val_metrics['recall_macro'].update(self.fake_report_0_1, self.real_report)
        self.val_metrics['f1_macro'].update(self.fake_report_0_1, self.real_report)
        
        # calculate the overall precision
        overall_precision = self.calculate_overall_precision(self.fake_report_0_1, self.real_report, batch_nmb)
        self.val_metrics['overall_precision'].append(overall_precision)
        
        # calculate the report cocnsistency loss
        val_loss_float = self.report_consistency_loss(self.fake_report, self.real_report)
        val_loss_0_1 = self.report_consistency_loss(self.fake_report_0_1, self.real_report)

        ##################################################
        # shuffle the val dataset since we need unpaired samples for below
        indices = torch.randperm(batch_nmb)
        self.real_report = self.real_report[indices]
        
        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)

        # generate valid and fake labels
        valid_img_sample = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake_img_sample = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))

        valid_report_sample = Tensor(np.ones((self.real_report.size(0), *self.report_discriminator.output_shape)))
        fake_report_sample = Tensor(np.zeros((self.real_report.size(0), *self.report_discriminator.output_shape)))

        # generate image
        self.fake_img = self.image_generator(z, self.real_report)
        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator(z, self.fake_report_0_1)
        
        ############ Log Loss for each step ##############
        gen_loss = self.generator_step(valid_img=valid_img_sample, valid_report=valid_report_sample)
        img_disc_loss = self.img_discriminator_step(valid_img_sample, fake_img_sample)
        report_disc_loss = self.report_discriminator_step(valid_report_sample, fake_report_sample)
        # also log the val_loss_float and val_loss_0_1
        self.log('val_loss_float', val_loss_float, on_step=True)
        self.log('val_loss_0_1', val_loss_0_1, on_step=True)
        ##################################################
        # Optional to log metrics on validation step
        if self.opt['trainer']['log_val_metrics_on_step']:

            self.log_images_on_cycle(batch_idx)
            self.log_reports_on_cycle(batch_idx)
            self.visualize_images(batch_idx)

            val_log_metrics = {
                'accuracy_micro' : self.val_metrics['accuracy_micro'].compute(),
                'precision_micro' : self.val_metrics['precision_micro'].compute(),
                'recall_micro' : self.val_metrics['recall_micro'].compute(),
                'f1_micro' : self.val_metrics['f1_micro'].compute(),
                'accuracy_macro' : self.val_metrics['accuracy_macro'].compute(),
                'precision_macro' : self.val_metrics['precision_macro'].compute(),
                'recall_macro' : self.val_metrics['recall_macro'].compute(),
                'f1_macro' : self.val_metrics['f1_macro'].compute(),
                'overall_precision' : torch.mean(torch.tensor(self.val_metrics['overall_precision']))
            }
            self.log_val_metrics(val_log_metrics, on_step=True)
        ###################################################
       
        
    def on_validation_epoch_end(self):
        # log the metrics
        val_log_metrics = {
                'accuracy_micro' : self.val_metrics['accuracy_micro'].compute(),
                'precision_micro' : self.val_metrics['precision_micro'].compute(),
                'recall_micro' : self.val_metrics['recall_micro'].compute(),
                'f1_micro' : self.val_metrics['f1_micro'].compute(),
                'accuracy_macro' : self.val_metrics['accuracy_macro'].compute(),
                'precision_macro' : self.val_metrics['precision_macro'].compute(),
                'recall_macro' : self.val_metrics['recall_macro'].compute(),
                'f1_macro' : self.val_metrics['f1_macro'].compute(),
                'overall_precision' : torch.mean(torch.tensor(self.val_metrics['overall_precision']))
        }
        self.log_val_metrics(val_log_metrics, on_step=False)
        # reset the metrics
        self.val_metrics['accuracy_micro'].reset()
        self.val_metrics['precision_micro'].reset()
        self.val_metrics['recall_micro'].reset()
        self.val_metrics['f1_micro'].reset()
        self.val_metrics['accuracy_macro'].reset()
        self.val_metrics['precision_macro'].reset()
        self.val_metrics['recall_macro'].reset()
        self.val_metrics['f1_macro'].reset()
        self.val_metrics['overall_precision'] = []


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

        self.logger.experiment.add_image(f"On step {self.phase} cycle img", cycle_img_tensor, step, dataformats='CHW')
        self.logger.experiment.add_image(f"On step {self.phase} real img", real_img_tensor, step, dataformats='CHW')
        self.logger.experiment.add_image(f"On step {self.phase} fake_img", fake_img_tensor, step, dataformats='CHW')

    def log_reports_on_cycle(self, batch_idx):
        real_report = self.real_report[0]
        cycle_report = self.cycle_report[0]
        # Process the generated report
        
        real_report = real_report.cpu().detach()
        real_report_raw = real_report.clone()
        
        real_report = torch.sigmoid(real_report)
        real_report = (real_report > 0.5).int()
        report_text_labels = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(real_report) if
                              val == 1]
        report_text_real = ', '.join(report_text_labels)
        # Convert tensor elements to strings for joining
        report_text_real_raw = ', '.join([str(item.item()) for item in real_report_raw])


        
        generated_report = cycle_report.cpu().detach()
        generated_report = torch.sigmoid(generated_report)
        generated_report_raw = generated_report.clone()
        generated_report = (generated_report > 0.5).int()
        report_text_labels_cycle = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(generated_report) if
                              val == 1]
        report_text_cycle = ', '.join(report_text_labels_cycle)
        # Convert tensor elements to strings for joining
        report_text_cycle_raw = ', '.join([str(item.item()) for item in generated_report_raw])


        step = self.current_epoch * batch_idx + batch_idx
        
        self.logger.experiment.add_text(f"On step {self.phase} cycle report", report_text_cycle, step)
        self.logger.experiment.add_text(f"On step {self.phase} real report", report_text_real , step)
        self.logger.experiment.add_text(f"On step {self.phase} cycle report raw", report_text_cycle_raw, step)
        self.logger.experiment.add_text(f"On step {self.phase} real report raw", report_text_real_raw, step)

    def save_images(self, batch_idx):
        # Process and save the real image
        real_image = self.convert_tensor_to_image(self.real_img[0])
        real_image_path = os.path.join(self.opt['trainer']['save_images_path'], f'real_image_{batch_idx}.png')
        real_image.save(real_image_path)

        # Process and save the cycle image
        cycle_image = self.convert_tensor_to_image(self.cycle_img[0])
        cycle_image_path = os.path.join(self.opt['trainer']['save_images_path'], f'cycle_image_{batch_idx}.png')
        cycle_image.save(cycle_image_path)

    def visualize_images(self, batch_idx):
        real_img = self.convert_tensor_to_image(self.real_img[0])
        plt.imshow(real_img)
        plt.axis('off')
        plt.show()
        cycle_img = self.convert_tensor_to_image(self.cycle_img[0])
        plt.imshow(cycle_img)
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

    def _get_report_generator(self):
        model_dict = {
            # 'ark' : ARKModel(num_classes=self.num_classes,
            # #                    ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK']),
            # 'biovil_t': BioViL(embedding_size=self.opt["report_generator"]["embedding_size"], 
            #                   num_classes=self.num_classes, 
            #                   hidden_1=self.opt["report_generator"]["classification_head_hidden1"],
            #                   hidden_2=self.opt["report_generator"]["classification_head_hidden2"], 
            #                   dropout_rate=self.opt["report_generator"]["dropout_prob"])
            'biovil_t': BioViL_V2()
        }

        return model_dict[self.opt['report_generator']['model']]

    def _get_report_discriminator(self):
        return ReportDiscriminator(input_dim=self.num_classes)
        #return CreativeDiscriminator(input_size=self.num_classes)
        #return VectorDiscriminator(input_size=self.num_classes, hidden_sizes=[128, 64, 32])

    def _get_image_generator(self):
        model_dict = {
            'cgan' : cGANconv(z_size=self.z_size, img_size=self.input_size, class_num=self.num_classes,
                    img_channels=self.opt["image_discriminator"]["channels"]) ,
            # 'ddpm' : DDPM(nn_model=ContextUnet(in_channels=3, n_feat=self.n_feat, n_classes=self.num_classes), 
            #               image_size=(self.input_size, self.input_size),
            #               betas=(float(self.opt['image_generator']['ddpm_beta1']), 
            #                      float(self.opt['image_generator']['ddpm_beta2'])),
            #               n_T=self.n_T,
            #               drop_prob=self.opt['image_generator']['ddpm_drop_prob']),
        }

        return model_dict[self.opt["image_generator"]["model"]]
        
    def _get_image_discriminator(self):
        return ImageDiscriminator(input_shape=(self.opt['image_discriminator']['channels'], 
                                               self.opt['image_discriminator']['img_height'],
                                               self.opt['image_discriminator']['img_width'])
                                 )
