import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.BioViL import BioViL
from models.ARK import ARKModel
from models.buffer import ReportBuffer, ImageBuffer
from models.Discriminator import ImageDiscriminator
from models.cGAN import cGAN
from losses.Test_loss import ClassificationLoss
from losses.Perceptual_loss import PerceptualLoss
from utils.environment_settings import env_settings
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
from torch.autograd import Variable
import numpy as np
import deepspeed as ds
import torchvision.transforms as transforms
from PIL import Image
import io
torch.autograd.set_detect_anomaly(True)

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

        # Define loss functions
        self.img_consistency_loss = PerceptualLoss()
        self.img_adversarial_loss = nn.MSELoss()
        self.report_consistency_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, img):
        # will be used in predict step for evaluation
        # img = img.float().to(self.device)
        # report = self.report_generator(img)
        # z = Variable(torch.randn_like(report)).to(self.device)
        # generated_img = self.image_generator(z, report)
        # return report, generated_img
        return img, None

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
    
    
    def generator_step(self, valid):
        # calculate loss for generator

        # adversarial loss
        adv_loss_IR = self.report_discriminator(self.fake_report) # return cosine similarity between fake report and entire dataset
        # adv_loss_IR = self.report_adv_criterion(self.report_discriminator(self.fake_report), valid)
        adv_loss_RI = self.img_adv_criterion(self.image_discriminator(self.fake_img), valid)
        # TODO : Should we really divide by 2?
        total_adv_loss = adv_loss_IR + adv_loss_RI
        ############################################################################################
        
        # cycle loss
        cycle_loss_IRI = self.img_consistency_criterion(self.real_img, self.cycle_img)
        cycle_loss_RIR = self.report_consistency_criterion(self.real_report, self.cycle_report)
        total_cycle_loss = self.lambda_cycle * (cycle_loss_IRI + cycle_loss_RIR)

        ############################################################################################

        total_gen_loss = total_adv_loss + total_cycle_loss

        ############################################################################################

        # Log losses
        metrics = {
            "gen_loss": total_gen_loss,
            "gen_adv_loss": total_adv_loss,
            "gen_cycle_loss": total_cycle_loss,
            "gen_adv_loss_IR": adv_loss_IR,
            "gen_adv_loss_RI": adv_loss_RI,
            "gen_cycle_loss_IRI": cycle_loss_IRI,
            "gen_cycle_loss_RIR": cycle_loss_RIR,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return total_gen_loss
    

    def discriminator_step(self, valid, fake):
        # fake_report = self.buffer_reports(self.fake_report)
        fake_img = self.buffer_images(self.fake_img)
        # calculate loss for discriminator
        ###########################################################################################
        # calculate on real data
        real_img_adv_loss = self.img_adv_criterion(self.image_discriminator(self.real_img), valid)
        # calculate on fake data
        fake_img_adv_loss = self.img_adv_criterion(self.image_discriminator(fake_img.detach()), fake)
        ###########################################################################################

        total_img_disc_loss = (real_img_adv_loss + fake_img_adv_loss) / 2
        metrics = {
            "img_disc_loss": total_img_disc_loss,
            "img_disc_adv_loss_real": real_img_adv_loss,
            "img_disc_adv_loss_fake": fake_img_adv_loss,
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return total_img_disc_loss



    def training_step(self, batch, batch_idx, optimizer_idx):
        self.real_img = batch['target']
        self.real_img = self.real_img.float()
        self.real_report = batch['report']

        z = Variable(torch.randn(self.batch_size, self.z_size)).to(self.device)
        
        # generate valid and fake labels
        valid = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))

        # generate fake reports and images
        self.fake_report = self.report_generator(self.real_img)
        self.fake_img = self.image_generator(z, self.real_report)

        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator(self.z, self.fake_report)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_loss = self.generator_step(valid)
            return gen_loss
        
        elif optimizer_idx == 2 or optimizer_idx == 3:
            disc_loss = self.discriminator_step(valid, fake)
            return disc_loss
        

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

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
    #         # and log using the logger (e.g., TensorBoard, WandB)
    #         generated_image = generated_images[i].cpu().detach()
    #         img_pil = transforms.ToPILImage()(generated_image.squeeze()).convert("RGB")
    #         img_buffer = io.BytesIO()
    #         img_pil.save(img_buffer, format="JPEG")
    #         img_buffer.seek(0)

    #         # Process the generated report
    #         generated_report = generated_reports[i].cpu().detach()
    #         generated_report = torch.sigmoid(generated_report)
    #         generated_report = (generated_report > 0.5).int()
    #         report_text_labels = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(generated_report) if val == 1]
    #         report_text = ', '.join(report_text_labels)

    #         # Log the image and the report text
    #         self.logger.experiment.add_image(f"Generated Image {i}", img_buffer, self.current_epoch, dataformats='HWC')
    #         self.logger.experiment.add_text(f"Generated Report {i}", report_text, self.current_epoch)

    
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
        return ClassificationLoss(env_settings.MASTER_LIST[self.data_imputation])
    
    def _get_image_generator(self):
        return cGAN(generator_layer_size=self.opt["image_generator"]["generator_layer_size"],
                    z_size=self.z_size,
                    img_size=self.input_size,
                    class_num=self.num_classes)
        
    def _get_image_discriminator(self):
        return ImageDiscriminator(input_shape=(self.opt['image_discriminator']['channels'], 
                                               self.opt['image_discriminator']['img_height'],
                                               self.opt['image_discriminator']['img_width'])
                                 )

    

        
        

