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
from models.ARK import ARKModel
from utils.buffer import ReportBuffer, ImageBuffer
from models.Discriminator import ImageDiscriminator, ReportDiscriminator
from models.cGAN import cGAN, cGANconv
from models.DDPM import ContextUnet, DDPM
from utils.environment_settings import env_settings
from losses.Losses import *
from losses.Metrics import *

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

    def initialize_components(self):
        """
        Initialize generators, discriminators, and buffers.
        """
        # Extract options
        self.data_imputation = self.opt['dataset']['data_imputation']
        self.input_size = self.opt["dataset"]["input_size"]
        self.num_classes = self.opt['trainer']['num_classes']
        self.buffer_size = self.opt['trainer']["buffer_size"]
        self.z_size = self.opt["image_generator"]["z_size"]
        self.lambda_cycle = self.opt['trainer']['lambda_cycle_loss']
        self.batch_size = self.opt["dataset"]["batch_size"]
        self.log_images_steps = self.opt["trainer"]["log_images_steps"]
        self.perceptual_loss_type = self.opt["image_generator"]["perceptual_loss"]
        self.report_discriminator_type = self.opt["report_discriminator"]["base"]
        self.n_feat = self.opt["image_generator"]["n_feat"]
        self.n_T = self.opt["image_generator"]["n_T"]


        # Initialize optimizer dictionary
        optimizer_dict = {
            'Adam': ds.ops.adam.FusedAdam, 
            'AdamW': torch.optim.AdamW,
        }

        # Initialize components
        self.report_generator = self._get_report_generator()
        self.image_generator = self._get_image_generator()
        self.report_discriminator = self._get_report_discriminator()
        self.image_discriminator = self._get_image_discriminator()
        self.buffer_reports = ReportBuffer(self.buffer_size)
        self.buffer_images = ImageBuffer(self.buffer_size)

        # Optimizers
        self.image_gen_optimizer = optimizer_dict[self.opt["image_generator"]["optimizer"]]
        self.report_gen_optimizer = optimizer_dict[self.opt["report_generator"]["optimizer"]]
        self.image_disc_optimizer = optimizer_dict[self.opt["image_discriminator"]["optimizer"]]
        if self.report_discriminator_type == "discriminator_network":
            self.report_disc_optimizer = optimizer_dict[self.opt["report_discriminator"]["optimizer"]]


    def define_loss_functions(self):
        """
        Define and initialize loss functions.
        """
        # Initialize loss functions
        if self.perceptual_loss_type == 'ark':
            self.img_consistency_loss = Loss(self.perceptual_loss_type, ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK'])
        else:
            self.img_consistency_loss = Loss(self.perceptual_loss_type)
        self.img_adversarial_loss = nn.MSELoss()

        self.report_consistency_loss = nn.BCEWithLogitsLoss()
        if self.report_discriminator_type == "discriminator_network":
            self.report_adversarial_loss = nn.MSELoss()

        self.MSE = nn.MSELoss()
    
    def forward(self, img):
        # will be used in predict step for evaluation
        img = img.float().to(self.device)
        report = self.report_generator(img).float()

        z = Variable(torch.randn(self.batch_size, self.z_size)).float().to(self.device)

        if self.opt['image_generator']['model'] == 'ddpm':
            generated_img = self.image_generator.generated_images(img.size(0), report)

        elif self.opt['image_generator']['model'] == 'cgan':
            generated_img = self.image_generator(z, report)
            
        return report, generated_img
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

        if self.report_discriminator_type == "discriminator_network":
            report_disc_opt_config = {
                     "lr" : self.opt["report_discriminator"]["learning_rate"],
                     "betas" : (self.opt["report_discriminator"]["beta1"], self.opt["report_discriminator"]["beta2"])
                 }
            report_discriminator_optimizer = self.report_disc_optimizer(
                     list(self.report_discriminator.parameters()),
                     **report_disc_opt_config,
                 )
            report_discriminator_scheduler = self.get_lr_scheduler(report_discriminator_optimizer, self.opt["report_discriminator"]["decay_epochs"])

        image_disc_opt_config = {
            "lr" : self.opt["image_discriminator"]["learning_rate"],
            "betas" : (self.opt["image_discriminator"]["beta1"], self.opt["image_discriminator"]["beta2"])
        }
        image_discriminator_optimizer = self.image_disc_optimizer(
            list(self.image_discriminator.parameters()),
            **image_disc_opt_config,
        )
        image_discriminator_scheduler = self.get_lr_scheduler(image_discriminator_optimizer, self.opt["image_discriminator"]["decay_epochs"])

        optimizers = [image_generator_optimizer, report_generator_optimizer, image_discriminator_optimizer]
        schedulers = [image_generator_scheduler, report_generator_scheduler, image_discriminator_scheduler]

        if self.report_discriminator_type == "discriminator_network":
            optimizers.append(report_discriminator_optimizer)
            schedulers.append(report_discriminator_scheduler)

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
    
    def report_adv_criterion(self, fake_report, real_report):
        # adversarial loss
        # additional discriminator network for reports (if needed)
        return self.report_adversarial_loss(fake_report, real_report)
    
    
    def generator_step(self, valid_img, valid_report):
        # calculate loss for generator

        # adversarial loss
        if self.report_discriminator_type == "discriminator_network":
            adv_loss_IR = self.report_adv_criterion(self.report_discriminator(self.fake_report), valid_report)
        elif self.report_discriminator_type == "cosine_similarity":
            adv_loss_IR = self.report_discriminator(self.fake_report) # return cosine similarity between fake report and entire dataset
        
        adv_loss_RI = self.img_adv_criterion(self.image_discriminator(self.fake_img), valid_img)
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

        # Log losses
        metrics = {
            "gen_predicted_noise_error" : self.fake_predicted_error,
            "gen_loss": total_gen_loss,
            "gen_adv_loss": total_adv_loss,
            "gen_cycle_loss": total_cycle_loss,
            "gen_adv_loss_IR": adv_loss_IR,
            "gen_adv_loss_RI": adv_loss_RI,
            "gen_cycle_predicted_noise_error" : self.cycle_predicted_error,
            "gen_cycle_loss_IRI": cycle_loss_IRI,
            "gen_cycle_loss_RIR": cycle_loss_RIR,
            "gen_cycle_loss_IR_MSE": cycle_loss_IRI_MSE,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return total_gen_loss


    def report_discriminator_step(self, valid, fake):
        fake_report = self.buffer_reports(self.fake_report)
        # calculate loss for discriminator
        real_report_adv_loss = self.report_adv_criterion(self.report_discriminator(self.real_report), valid)
        # calculate on fake data
        fake_report_adv_loss = self.report_adv_criterion(self.report_discriminator(fake_report.detach()), fake)
        total_report_disc_loss = (real_report_adv_loss + fake_report_adv_loss) / 2

        metrics = {
            "report_disc_loss": total_report_disc_loss,
            "report_disc_adv_loss_real": real_report_adv_loss,
            "report_disc_adv_loss_fake": fake_report_adv_loss,
        }

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return total_report_disc_loss


    def image_discriminator_step(self, valid, fake):
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
        
        total_img_disc_loss = (real_img_adv_loss + fake_img_adv_loss) / 2 * 0.1
        # print(f'disc_total:{total_img_disc_loss}')
        
        metrics = {
            "img_disc_loss": total_img_disc_loss,
            "img_disc_adv_loss_real": real_img_adv_loss,
            "img_disc_adv_loss_fake": fake_img_adv_loss,
        }

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return total_img_disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        # create noise variables
        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)
        noise = torch.randn_like(self.real_img).to(self.device)

        # generate valid and fake labels
        valid_img_sample = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake_img_sample = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape))) 

        
        valid_report_sample = Tensor(np.ones((self.real_report.size(0), *self.report_discriminator.output_shape)))
        fake_report_sample = Tensor(np.zeros((self.real_report.size(0), *self.report_discriminator.output_shape)))


        self.fake_report = self.report_generator(self.real_img)


        if self.opt['image_generator']['model'] == 'ddpm':
            self.fake_predicted_error = self.image_generator(self.real_img, noise, self.real_report)
            self.fake_img = self.image_generator.generate_image(batch_size=batch_nmb, condition=self.real_report)

        elif self.opt['image_generator']['model'] == 'cgan':
            self.fake_img = self.image_generator(z, self.real_report)


        fake_reports = torch.sigmoid(self.fake_report)
        fake_reports = torch.where(fake_reports < 0.5, 0, fake_reports)
        fake_reports = torch.where(fake_reports >= 0.5, 1, fake_reports)


        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)

        # TODO : Update ddpm class input params ( x  and c )
        if self.opt['image_generator']['model'] == 'ddpm':
            self.cycle_predicted_error = self.image_generator(self.real_img, noise, self.fake_reports)
            self.cycle_img = self.image_generator.generate_image(batch_size=batch_nmb, condition=self.fake_reports)
        else:
            self.cycle_img = self.image_generator(z, fake_reports)
        

        if (batch_idx % self.log_images_steps) == 0 and optimizer_idx == 0:
            self.log_images_on_cycle(batch_idx)
            self.log_reports_on_cycle(batch_idx)
            self.visualize_images(batch_idx)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_loss = self.generator_step(valid_img=valid_img_sample, valid_report=valid_report_sample)
            return {"loss": gen_loss}
        
        elif optimizer_idx == 2 or optimizer_idx == 3:
            img_disc_loss = self.image_discriminator_step(valid_img_sample, fake_img_sample)
            if self.report_discriminator_type == "discriminator_network":
                report_disc_loss = self.report_discriminator_step(valid_report_sample, fake_report_sample)
                return {"loss": img_disc_loss, "report_disc_loss": report_disc_loss}
            else:
                return {"loss": img_disc_loss}
        

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        # return generated report and generated image from generated report
        report, image = self(batch['target'])
        return report, image
    

    def on_validation_epoch_end(self):
        # Select a small number of validation samples
        num_samples = min(5, self.batch_size)
        val_samples = next(iter(self.val_dataloader))
    
        # Generate reports and images for these samples
        generated_reports, generated_images = self(val_samples['target'])
    
        # Log the generated reports and images
        for i in range(num_samples):
            # Convert the tensor to a suitable image format (e.g., PIL Image)
            generated_image = generated_images[i].cpu().detach()
            img_pil = transforms.ToPILImage()(generated_image.squeeze()).convert("RGB")
    
            # Convert PIL Image back to tensor
            img_tensor = to_tensor(img_pil)
    
            # Process the generated report
            generated_report = generated_reports[i].cpu().detach()
            generated_report = torch.sigmoid(generated_report)
            generated_report = (generated_report > 0.5).int()
            report_text_labels = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(generated_report) if val == 1]
            report_text = ', '.join(report_text_labels)
    
            # Log the image and the report text
            self.logger.experiment.add_image(f"Generated Image {i}", img_tensor, self.current_epoch, dataformats='CHW')
            self.logger.experiment.add_text(f"Generated Report {i}", report_text, self.current_epoch)


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
        model_name = self.opt["report_generator"]["image_encoder_model"]
        if model_name == "Ark":
            return ARKModel(num_classes=self.num_classes,
                            ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK'])
        
        elif model_name == "BioVil":
            return BioViL(embedding_size=self.opt["report_generator"]["embedding_size"], 
                          num_classes=self.num_classes, 
                          hidden_1=self.opt["report_generator"]["classification_head_hidden1"],
                          hidden_2=self.opt["report_generator"]["classification_head_hidden2"], 
                          dropout_rate=self.opt["report_generator"]["dropout_prob"])
        else:
            raise NotImplementedError(f"Model {model_name} not implemented for report generation.")
        

    def _get_report_discriminator(self):
        if self.report_discriminator_type == "discriminator_network":
            return ReportDiscriminator(input_dim=self.num_classes)
        elif self.report_discriminator_type == "cosine_similarity":
            return Loss("classification", env_settings.MASTER_LIST[self.data_imputation])
        else:
            raise NotImplementedError(f"Model {self.report_discriminator_type} not implemented for report discriminator.")
    
    def _get_image_generator(self):
        # return cGAN(generator_layer_size=self.opt["image_generator"]["generator_layer_size"], z_size=self.z_size,
        #               img_size=self.input_size, class_num=self.num_classes)

        
        models = {
            'cgan' : cGANconv(z_size=self.z_size, img_size=self.input_size, class_num=self.num_classes,
                    img_channels=self.opt["image_discriminator"]["channels"]) ,

            'ddpm' : DDPM(nn_model=ContextUnet(in_channels=3, n_feat=self.n_feat, n_classes=self.num_classes), 
                          image_size=(self.input_size, self.input_size),
                          betas=(self.opt['image_generator'['ddpm_beta1'], self.opt['image_generator']['ddpm_beta2']]),
                          n_T=self.n_T,
                          drop_prob=self.opt['image_generator']['ddpm_drop_prob'])
        }

        return models[self.opt["image_generator"]["model"]]
        
    def _get_image_discriminator(self):
        return ImageDiscriminator(input_shape=(self.opt['image_discriminator']['channels'], 
                                               self.opt['image_discriminator']['img_height'],
                                               self.opt['image_discriminator']['img_width'])
                                 )