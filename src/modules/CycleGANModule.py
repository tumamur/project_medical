from typing import Union, List

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.BioViL import BioViL, BioViL_V2
from models.ARK import ARKModel
from models.buffer import ReportBuffer, ImageBuffer
from models.Discriminator import ImageDiscriminator
from models.cGAN import cGAN, cGANconv
from losses.Test_loss import ClassificationLoss
from losses.Perceptual_loss import PerceptualLoss
from models.DDPM import ContextUnet, DDPM
from losses.Perceptual_xrays import Perceptual_xray
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from utils.environment_settings import env_settings
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
from torch.autograd import Variable
import numpy as np
import deepspeed as ds
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor
import io
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Precision, Recall, F1Score
import os
from utils.environment_settings import env_settings


class CycleGAN(pl.LightningModule):

    """
    Generator for image-to-report translation (report_generator).
    Generator for report-to-image translation (image_generator).
    Discriminator for Generated Reports (report_dicriminator). --> Currenlty using ClassificationLoss (Cosine Similarity)
    Discriminator for Generated Images (image_generator).
        
    """

    def __init__(self, opt, val_dataloader):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.save_hyperparameters(opt)

        self.data_imputation = opt['dataset']['data_imputation']
        self.input_size = opt["dataset"]["input_size"]
        self.num_classes = opt['trainer']['num_classes']
        self.n_epochs = opt['trainer']['n_epoch']
        self.buffer_size = opt['trainer']["buffer_size"]
        self.lambda_cycle = opt['trainer']['lambda_cycle_loss']
        self.val_dataloader = val_dataloader
        self.batch_size = opt["dataset"]["batch_size"]
        self.z_size = opt["image_generator"]["z_size"]
        self.log_images_steps = opt["trainer"]["log_images_steps"]
        self.MSE = nn.MSELoss()
        self.n_feat = self.opt["image_generator"]["n_feat"]
        self.n_T = self.opt["image_generator"]["n_T"]
        self.save_images = opt["trainer"]["save_images"]
        if self.save_images:
            self.save_path = env_settings.SAVE_IMAGES_PATH
            print(f'Saving images to {self.save_path}')
        self.visualize = opt["trainer"]["visualize"]
        # Initialize optimizers
        optimizer_dict = {
            'Adam': ds.ops.adam.FusedAdam, 
            'AdamW': torch.optim.AdamW,
        }
        self.image_gen_optimizer = optimizer_dict[opt["image_generator"]["optimizer"]]
        self.image_disc_optimizer = optimizer_dict[opt["image_discriminator"]["optimizer"]]
        self.report_gen_optimizer = optimizer_dict[opt["report_generator"]["optimizer"]]

        # Define components of the GAN
        self.report_generator = self._get_report_generator()
        self.report_discriminator = self._get_report_discriminator()
        self.buffer_reports = ReportBuffer(self.buffer_size)
        self.image_generator = self._get_image_generator()
        self.image_discriminator = self._get_image_discriminator()
        self.buffer_images = ImageBuffer(self.buffer_size)
        self.gen_threshold = opt["trainer"]["gen_training_threshold"]
        self.disc_threshold = opt["trainer"]["disc_training_threshold"]

        # Define loss functions
        # self.img_consistency_loss = PerceptualLoss()
        self.img_consistency_loss = Perceptual_xray()
        self.img_adversarial_loss = nn.MSELoss()
        self.report_consistency_loss = nn.BCEWithLogitsLoss()

        # Metrics
        # Possibly change to average="macro"
        self.val_accuracy_micro = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_accuracy_macro = Accuracy(task="multilabel", average="macro", num_labels=self.num_classes)
        self.val_precision_micro = Precision(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_precision_macro = Precision(task="multilabel", average="macro", num_labels=self.num_classes)
        self.val_recall_micro = Recall(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_recall_macro = Recall(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_f1_micro = F1Score(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_f1_macro = F1Score(task="multilabel", average="macro", num_labels=self.num_classes)
        self.test_accuracy_cycle_micro = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes)
        self.test_recall_cycle_micro = Recall(task="multilabel", average="micro", num_labels=self.num_classes)
        self.test_f1_cycle_micro = F1Score(task="multilabel", average="micro", num_labels=self.num_classes)
        self.test_precision_cycle_micro = Precision(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_accuracy_cycle_micro = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_precision_cycle_micro = Precision(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_recall_cycle_micro = Recall(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_f1_cycle_micro = F1Score(task="multilabel", average="micro", num_labels=self.num_classes)
        self.val_accuracy_cycle_macro = Accuracy(task="multilabel", average="macro", num_labels=self.num_classes)
        self.val_precision_cycle_macro = Precision(task="multilabel", average="macro", num_labels=self.num_classes)
        self.val_recall_cycle_macro = Recall(task="multilabel", average="macro", num_labels=self.num_classes)
        self.val_f1_cycle_macro = F1Score(task="multilabel", average="macro", num_labels=self.num_classes)


    def forward(self, img):
       
        # will be used in predict step for evaluation
        img = img.float().to(self.device)
        report = self.report_generator(img).float()
        
        #z = Variable(torch.randn(self.batch_size, self.z_size)).float().to(self.device)
        

        #generated_img = self.image_generator(z, report)
        return None
        #return img, None

    def get_lr_scheduler(self, optimizer, decay_epochs):
        def lr_lambda(epoch):
            len_decay_phase = self.n_epochs - decay_epochs + 1.0
            curr_decay_step = max(0, epoch - decay_epochs + 1.0)
            val = 1.0 - curr_decay_step / len_decay_phase
            return max(0.0, val)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        image_gen_opt_config = {
            "lr" : self.opt["image_generator"]["learning_rate"],
            "betas" : (self.opt["image_generator"]["beta1"], self.opt["image_generator"]["beta2"])
        }
        image_generator_optimizer = self.image_gen_optimizer(
            list(self.image_generator.parameters()),
            **image_gen_opt_config,
        )
        image_generator_scheduler = self.get_lr_scheduler(image_generator_optimizer, self.opt["image_generator"]["decay_epochs"])


        report_gen_opt_config = {
            "lr" : self.opt["report_generator"]["learning_rate"],
            "betas" : (self.opt["report_generator"]["beta1"], self.opt["report_generator"]["beta2"])
        }
        report_generator_optimizer = self.report_gen_optimizer(
            list(self.report_generator.parameters()),
            **report_gen_opt_config,
        )
        report_generator_scheduler = self.get_lr_scheduler(report_generator_optimizer, self.opt["report_generator"]["decay_epochs"])


        image_disc_opt_config = {
            "lr" : self.opt["image_discriminator"]["learning_rate"],
            "betas" : (self.opt["image_discriminator"]["beta1"], self.opt["image_discriminator"]["beta2"])
        }
        image_discriminator_optimizer = self.image_disc_optimizer(
            list(self.image_discriminator.parameters()),
            **image_disc_opt_config,
        )
        image_discriminator_scheduler = self.get_lr_scheduler(image_discriminator_optimizer, self.opt["image_discriminator"]["decay_epochs"])

        # optimizers = [image_generator_optimizer, report_generator_optimizer image_discriminator_optimizer, report_discriminator_optimizer]
        optimizers = [image_generator_optimizer, report_generator_optimizer, image_discriminator_optimizer]
        schedulers = [image_generator_scheduler, report_generator_scheduler, image_discriminator_scheduler]
        return optimizers, schedulers


    def img_adv_criterion(self, fake_image, real_image):
        # adversarial loss
        return self.img_adversarial_loss(fake_image, real_image)

    def img_consistency_criterion(self, real_image, cycle_image):
        # reconstruction loss
        return self.img_consistency_loss(real_image, cycle_image)
    
    def report_consistency_criterion(self, real_report, cycle_report):
        # reconstruction loss
        return self.report_consistency_loss(real_report, cycle_report)

    def generator_step(self, valid, mode="train"):
        # calculate loss for generator

        # adversarial loss
        adv_loss_IR = self.report_discriminator(self.fake_report) # return cosine similarity between fake report and entire dataset
        # adv_loss_IR = self.report_adv_criterion(self.report_discriminator(self.fake_report), valid)
        adv_loss_RI = self.img_adv_criterion(self.image_discriminator(self.fake_img), valid)
        # TODO : Should we really divide by 2?
        total_adv_loss = adv_loss_IR + adv_loss_RI
        # print(f'adv_loss:{total_adv_loss}')
        ############################################################################################
        
        # cycle loss
        cycle_loss_IRI = self.img_consistency_criterion(self.real_img, self.cycle_img)
        cycle_loss_IRI_MSE = self.MSE(self.real_img, self.cycle_img)
        # print(f'cycle_loss_IRI:{cycle_loss_IRI}')
        cycle_loss_RIR = self.report_consistency_criterion(self.cycle_report, self.real_report)
        # print(f'cycle_loss_RIR:{cycle_loss_RIR}')
        total_cycle_loss = self.lambda_cycle * (cycle_loss_IRI + cycle_loss_RIR) + 1 * cycle_loss_IRI_MSE

        ############################################################################################

        total_gen_loss = total_adv_loss + total_cycle_loss
        # print(f'gen_loss:{total_gen_loss}')
        ############################################################################################

        if mode == "train":
            # Log losses
            metrics = {
                "gen_loss": total_gen_loss,
                "gen_adv_loss": total_adv_loss,
                "gen_cycle_loss": total_cycle_loss,
                "gen_adv_loss_IR": adv_loss_IR,
                "gen_adv_loss_RI": adv_loss_RI,
                "gen_cycle_loss_IRI": cycle_loss_IRI,
                "gen_cycle_loss_RIR": cycle_loss_RIR,
                "gen_cycle_loss_IR_MSE": cycle_loss_IRI_MSE,
            }
        elif mode == "validation":
            # Log losses
            metrics = {
                "val_gen_loss": total_gen_loss,
                "val_gen_adv_loss": total_adv_loss,
                "val_gen_cycle_loss": total_cycle_loss,
                "val_gen_adv_loss_IR": adv_loss_IR,
                "val_gen_adv_loss_RI": adv_loss_RI,
                "val_gen_cycle_loss_IRI": cycle_loss_IRI,
                "val_gen_cycle_loss_RIR": cycle_loss_RIR,
                "val_gen_cycle_loss_IR_MSE": cycle_loss_IRI_MSE,
            }

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return total_gen_loss

    def discriminator_step(self, valid, fake, mode="train"):
        # fake_report = self.buffer_reports(self.fake_report)
        fake_img = self.buffer_images(self.fake_img)
        # calculate loss for discriminator
        ###########################################################################################
        # calculate on real data
        real_img_adv_loss = self.img_adv_criterion(self.image_discriminator(self.real_img), valid)
        # calculate on fake data
        fake_img_adv_loss = self.img_adv_criterion(self.image_discriminator(fake_img.detach()), fake)
        ###########################################################################################
        # print(f'disc_real_adv:{real_img_adv_loss}')
        # print(f'disc_fake_adv:{fake_img_adv_loss}')
        
        total_img_disc_loss = (real_img_adv_loss + fake_img_adv_loss) / 2
        # print(f'disc_total:{total_img_disc_loss}')

        if mode == "train":
            metrics = {
                "img_disc_loss": total_img_disc_loss,
                "img_disc_adv_loss_real": real_img_adv_loss,
                "img_disc_adv_loss_fake": fake_img_adv_loss,
            }
        elif mode == "validation":
            metrics = {
                "val_img_disc_loss": total_img_disc_loss,
                "val_img_disc_adv_loss_real": real_img_adv_loss,
                "val_img_disc_adv_loss_fake": fake_img_adv_loss,
            }

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return total_img_disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)
        
        # generate valid and fake labels
        valid = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))

        # generate fake reports and images
        self.fake_report = self.report_generator(self.real_img)
        self.fake_img = self.image_generator(z, self.real_report)

        fake_reports = torch.sigmoid(self.fake_report)
        fake_reports = torch.where(fake_reports > 0.5, 1, 0)

        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator(z, fake_reports)

        cycle_reports = torch.sigmoid(self.cycle_report)
        cycle_reports = torch.where(cycle_reports > 0.5, 1.0, 0.0)
        self.test_accuracy_cycle_micro.update(cycle_reports, self.real_report)
        self.test_recall_cycle_micro.update(cycle_reports, self.real_report)
        self.test_f1_cycle_micro.update(cycle_reports, self.real_report)
        self.test_precision_cycle_micro.update(cycle_reports, self.real_report)
        test_precision_overall = self.calculate_metrics_overall(cycle_reports, self.real_report, batch_nmb)

        test_metrics = {
            "test_accuracy_cycle_micro": self.test_accuracy_cycle_micro,
            "test_recall_cycle_micro": self.test_recall_cycle_micro,
            "test_f1_cycle_micro": self.test_f1_cycle_micro,
            "test_precision_cycle_micro": self.test_precision_cycle_micro,
            "test_precision_overall_cycle": test_precision_overall,
        }

        self.log_dict(test_metrics, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % 1000 == 0:
            if optimizer_idx == 0:
                self.log_images_on_cycle(batch_idx)
                self.log_reports_on_cycle(batch_idx)
                self.visualize_images(batch_idx)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_test_loss = self.generator_step(valid)
            if gen_test_loss > self.gen_threshold:
                gen_loss = gen_test_loss
                return gen_loss

        elif optimizer_idx == 2 or optimizer_idx == 3:
            disc_test_loss = self.discriminator_step(valid, fake)
            if disc_test_loss > self.disc_threshold:
                disc_loss = disc_test_loss
                return disc_loss

    def validation_step(self, batch, batch_idx):
        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        self.fake_report = self.report_generator(self.real_img)
        self.fake_report = torch.sigmoid(self.fake_report)
        self.fake_report_0_1 = torch.where(self.fake_report > 0.5, 1.0, 0.0)

        # self.val_accuracy.update(self.fake_report_0_1, self.real_report)
        self.val_accuracy_micro.update(self.fake_report_0_1, self.real_report)
        self.val_accuracy_macro.update(self.fake_report_0_1, self.real_report)
        self.val_precision_micro.update(self.fake_report_0_1, self.real_report)
        self.val_recall_macro.update(self.fake_report_0_1, self.real_report)
        self.val_recall_micro.update(self.fake_report_0_1, self.real_report)
        self.val_recall_macro.update(self.fake_report_0_1, self.real_report)
        self.val_f1_macro.update(self.fake_report_0_1, self.real_report)
        self.val_f1_micro.update(self.fake_report_0_1, self.real_report)
        precision_overall = self.calculate_metrics_overall(self.fake_report_0_1, self.real_report, batch_nmb)

        val_loss_float = self.report_consistency_loss(self.fake_report, self.real_report)
        val_loss_0_1 = self.report_consistency_loss(self.fake_report_0_1, self.real_report)

        # Shuffle data here as paired data is needed for above part
        indices = torch.randperm(batch_nmb)
        self.real_report = self.real_report[indices]
        # Calculating losses for CycleGAN
        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)

        # generate valid and fake labels
        valid = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))

        # generate and images
        self.fake_img = self.image_generator(z, self.real_report)

        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator(z, self.fake_report_0_1)

        cycle_report = torch.sigmoid(self.cycle_report)
        cycle_report = torch.where(cycle_report > 0.5, 1.0, 0.0)
        precision_overall_cycle = self.calculate_metrics_overall(cycle_report, self.real_report, batch_nmb)
        self.val_accuracy_cycle_micro.update(cycle_report, self.real_report)
        self.val_accuracy_cycle_macro.update(cycle_report, self.real_report)
        self.val_precision_cycle_micro.update(cycle_report, self.real_report)
        self.val_precision_cycle_macro.update(cycle_report, self.real_report)
        self.val_recall_cycle_micro.update(cycle_report, self.real_report)
        self.val_recall_cycle_macro.update(cycle_report, self.real_report)
        self.val_f1_cycle_micro.update(cycle_report, self.real_report)
        self.val_f1_cycle_macro.update(cycle_report, self.real_report)

        val_gen_loss = self.generator_step(valid, "validation")
        val_disc_loss = self.discriminator_step(valid, fake, "validation")

        val_metrics = {
            "val_accuracy_micro": self.val_accuracy_micro,
            "val_accuracy_macro": self.val_accuracy_macro,
            "val_precision_micro": self.val_precision_micro,
            "val_precision_macro": self.val_precision_macro,
            "val_precision_overall": precision_overall,
            "val_precision_overall_cycle": precision_overall_cycle,
            "val_recall_micro": self.val_recall_micro,
            "val_recall_macro": self.val_recall_macro,
            "val_f1_micro": self.val_f1_micro,
            "val_f1_macro": self.val_f1_macro,
            "val_loss_sigmoid": val_loss_float,
            "val_loss_where": val_loss_0_1,
            "val_accuracy_cycle_micro": self.val_accuracy_cycle_micro,
            "val_accuracy_cycle_macro": self.val_accuracy_cycle_macro,
            "val_precision_cycle_micro": self.val_precision_cycle_micro,
            "val_precision_cycle_macro": self.val_precision_cycle_macro,
            "val_recall_cycle_micro": self.val_recall_cycle_micro,
            "val_recall_cycle_macro": self.val_recall_cycle_macro,
            "val_f1_cycle_micro": self.val_f1_cycle_micro,
            "val_f1_cycle_macro": self.val_f1_macro,
        }

        self.log_dict(val_metrics, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    '''def validation_epoch_end(self, validation_step_outputs):
        self.test_recall_cycle_micro.reset()
        self.test_f1_cycle_micro.reset()
        self.test_accuracy_cycle_micro.reset()
        self.test_precision_cycle_micro.reset()
        self.val_accuracy_micro.reset()
        self.val_accuracy_macro.reset()
        self.val_precision_micro.reset()
        self.val_precision_macro.reset()
        self.val_recall_micro.reset()
        self.val_recall_macro.reset()
        self.val_f1_micro.reset()
        self.val_f1_macro.reset()
        self.val_f1_cycle_micro.reset()
        self.val_f1_cycle_macro.reset()
        self.val_accurracy_cycle_micro.reset()
        self.val_accuracy_cycle_macro.reset()
        self.val_precision_cycle_micro.reset()
        self.val_precision_cycle_macro.reset()
        self.val_recall_cycle_micro.reset()
        self.val_recall_cycle_macro.reset()'''

    def predict_step(self, batch, batch_idx):
        # return generated report and generated image from generated report
        report, image = self(batch['target'])
        return report, image

    # def on_validation_epoch_end(self):
    #     # Select a small number of validation samples
    #     num_samples = min(5, self.batch_size)
    #     val_samples = next(iter(self.val_dataloader))
    
    #     # Generate reports and images for these samples
    #     generated_reports, generated_images = self(val_samples['target'])
    
    #     # Log the generated reports and images
    #     for i in range(num_samples):
    #         # Convert the tensor to a suitable image format (e.g., PIL Image)
    #         generated_image = generated_images[i].cpu().detach()
    #         img_pil = transforms.ToPILImage()(generated_image.squeeze()).convert("RGB")
    
    #         # Convert PIL Image back to tensor
    #         img_tensor = to_tensor(img_pil)
    
    #         # Process the generated report
    #         generated_report = generated_reports[i].cpu().detach()
    #         generated_report = torch.sigmoid(generated_report)
    #         generated_report = (generated_report > 0.5).int()
    #         report_text_labels = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(generated_report) if val == 1]
    #         report_text = ', '.join(report_text_labels)
    
    #         # Log the image and the report text
    #         self.logger.experiment.add_image(f"Generated Image {i}", img_tensor, self.current_epoch, dataformats='CHW')
    #         self.logger.experiment.add_text(f"Generated Report {i}", report_text, self.current_epoch)


    def calculate_metrics_overall(self, preds, targets, batch_nmb):
        exact_matches = torch.all(preds == targets, dim=1)
        true_positives = torch.sum(exact_matches).item()

        precision = true_positives / batch_nmb

        return precision

    def log_images_on_cycle(self, batch_idx):

        cycle_img_1 = self.cycle_img[0]
        real_img_1 = self.real_img[0]
        fake_img_1 = self.fake_img[0]

        #cycle_img_pil = transforms.ToPILImage()(cycle_img_1.squeeze()).convert("RGB")
        #real_img_pil = transforms.ToPILImage()(real_img_1.squeeze()).convert('RGB')
        #fake_img_pil = transforms.ToPILImage()(fake_img_1.squeeze()).convert("RGB")

        #cycle_img_tensor = to_tensor(cycle_img_pil)
        #real_img_tensor = to_tensor(real_img_pil)
        #fake_img_tensor = to_tensor(fake_img_pil)
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
        #tensor = self.real_img[0]

        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        #denorm = tensor.clone().cpu().detach()
        #for t, m, s in zip(denorm, mean, std):
         #   t.mul_(s).add_(m)

        #image_to_display = denorm.numpy().transpose(1, 2, 0)
        #image_to_display = np.clip(image_to_display, 0, 1)
        # plt.imshow(tensor.permute(1, 2, 0).cpu().detach())
        real_img = self.convert_tensor_to_image(self.real_img[0])
        plt.imshow(real_img)
        plt.axis('off')
        plt.show()

        #cycle_tensor = self.cycle_img[0]
        #denorm = cycle_tensor.clone().cpu().detach()
        #for t, m, s in zip(denorm, mean, std):
         #   t.mul_(s).add_(m)
        #image_to_display = denorm.numpy().transpose(1, 2, 0)
        #image_to_display = np.clip(image_to_display, 0, 1)
        # plt.imshow(cycle_tensor.permute(1, 2, 0).cpu().detach())
        cycle_img = self.convert_tensor_to_image(self.cycle_img[0])
        plt.imshow(cycle_img)
        plt.axis('off')
        plt.show()

    def save_images(self, batch_idx):
        # Create the folder if it does not exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Process and save the real image
        real_image = self.convert_tensor_to_image(self.real_img[0])
        real_image_path = os.path.join(self.save_folder, f'real_image_{batch_idx}.png')
        real_image.save(real_image_path)

        # Process and save the cycle image
        cycle_image = self.convert_tensor_to_image(self.cycle_img[0])
        cycle_image_path = os.path.join(self.save_folder, f'cycle_image_{batch_idx}.png')
        cycle_image.save(cycle_image_path)

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
        model_name = self.opt["report_generator"]["image_encoder_model"]
        if model_name == "ARK":
            return ARKModel(num_classes=self.num_classes,
                            ark_pretrained_path=env_settings.PRETRAINED_PATH_ARK)
        
        elif model_name == "BioVil":
            #return BioViL(embedding_size=self.opt["report_generator"]["embedding_size"],
             #             num_classes=self.num_classes,
              #            hidden_1=self.opt["report_generator"]["classification_head_hidden1"],
               #           hidden_2=self.opt["report_generator"]["classification_head_hidden2"],
                #          dropout_rate=self.opt["report_generator"]["dropout_prob"])
            return BioViL_V2()
        else:
            raise NotImplementedError(f"Model {model_name} not implemented for report generation.")

    def _get_report_discriminator(self):
        return ClassificationLoss(env_settings.MASTER_LIST[self.data_imputation])
    
    def _get_image_generator(self):
        # return cGAN(generator_layer_size=self.opt["image_generator"]["generator_layer_size"],
          #          z_size=self.z_size,
           #         img_size=self.input_size,
            #        class_num=self.num_classes)
        return cGANconv(z_size=self.z_size, img_size=self.input_size, class_num=self.num_classes,
                           img_channels=self.opt["image_discriminator"]["channels"])
        # return DDPM(nn_model=ContextUnet(in_channels=3, n_feat=self.n_feat, n_classes=self.num_classes),
          #                betas=(float(self.opt['image_generator']['ddpm_beta1']), float(self.opt['image_generator']['ddpm_beta2'])),
           #               n_T=self.n_T, drop_prob=self.opt['image_generator']['ddpm_drop_prob'])
        
    def _get_image_discriminator(self):
        return ImageDiscriminator(input_shape=(self.opt['image_discriminator']['channels'], 
                                               self.opt['image_discriminator']['img_height'],
                                               self.opt['image_discriminator']['img_width'])
                                 )