import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from models.BioViL import BioViL
from models.ARK import ARKModel
from models.buffer import ReportBuffer, ImageBuffer
from models.Discriminator import ReportDiscriminator, ImageDiscriminator
from losses.Metrics import Metrics
from losses.CombinationLoss import CombinationLoss
from losses.Test_loss import ClassificationLoss
from losses.Perceptual_loss import PerceptualLoss
from torch.optim import lr_scheduler
from utils.environment_settings import env_settings
import torchmetrics.functional as F
from losses.ClassificationLoss import SimilarityLoss, AdversarialLoss
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
from torch.autograd import Variable
import numpy as np
import deepspeed as ds
import torchvision.transforms as transforms
from PIL import Image
import io
from models.imagen_pytorch.imagen_pytorch import ImagenTrainer

torch.autograd.set_detect_anomaly(True)

class CycleGAN(pl.LightningModule):

    """
    Generator for image-to-report translation (report_generator).
    Generator for report-to-image translation (image_generator).
    Discriminator for Generated Reports (report_dicriminator). --> Currenlty using ClassificationLoss (Cosine Similarity)
    Discriminator for Generated Images (image_generator).
        
    """

    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.save_hyperparameters(opt)
        # self.automatic_optimization = False
        self.data_imputation = opt['dataset']['data_imputation']
        self.num_classes = opt['trainer']['num_classes']
        self.n_epochs = opt['trainer']['n_epoch']
        self.buffer_size = opt['trainer']["buffer_size"]
        self.lambda_cycle = opt['trainer']['lambda_cycle_loss']

        optimizer_dict = {
            'Adam': ds.ops.adam.FusedAdam, 
            'AdamW': torch.optim.AdamW,
        }
        self.image_gen_optimizer = optimizer_dict[self.opt["image_generator"]["optimizer"]]
        self.image_disc_optimizer = optimizer_dict[self.opt["image_discriminator"]["optimizer"]]
        self.report_gen_optimizer = optimizer_dict[self.opt["report_generator"]["optimizer"]]

        # Define Report Generation Component
        self.report_generator = self._get_report_generator(self.opt['report_generator']['image_encoder_model'])
        self.report_discriminator = self._get_report_discriminator()
        self.buffer_reports = ReportBuffer(self.buffer_size)
        
        # Define Image Generation Component
        self.image_generator = self._get_image_generator(self.opt['image_generator']['report_encoder_model'])
        self.image_discriminator = self._get_image_discriminator()
        self.buffer_images = ImageBuffer(self.buffer_size)

        self.img_consistency_loss = PerceptualLoss()
        self.img_adversarial_loss = nn.MSELoss()
        # self.report_adversarial_loss = nn.MSELoss()
        # self.report_consistency_loss = ClassificationLoss(env_settings.MASTER_LIST[self.data_imputation])
        self.report_consistency_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, img):
        # will be used in predict step for evaluation
        img = img.float()
        report = self.report_generator(img)
        generated_img = self.image_generator.sample(report)
        return report, generated_img
    
    def init_weights(self):
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for net in [self.report_generator, self.image_generator, self.image_discriminator]:
            net.apply(init_fn)

    def setup(self, stage):
        if stage == "fit":
            self.init_weights()
            print("Model initialized.")

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
        image_generator_optimizer = self.optimizer(
            list(self.image_generator.parameters()),
            **image_gen_opt_config,
        )
        image_generator_scheduler = self.get_lr_scheduler(image_generator_optimizer, self.opt["image_generator"]["decay_epochs"])


        report_gen_opt_config = {
            "lr" : self.opt["report_generator"]["learning_rate"],
            "betas" : (self.opt["report_generator"]["beta1"], self.opt["report_generator"]["beta2"])
        }
        report_generator_optimizer = self.optimizer(
            list(self.report_generator.parameters()),
            **report_gen_opt_config,
        )
        report_generator_scheduler = self.get_lr_scheduler(report_generator_optimizer, self.opt["report_generator"]["decay_epochs"])


        image_disc_opt_config = {
            "lr" : self.opt["image_discriminator"]["learning_rate"],
            "betas" : (self.opt["image_discriminator"]["beta1"], self.opt["image_discriminator"]["beta2"])
        }
        image_discriminator_optimizer = self.optimizer(
            list(self.image_discriminator.parameters()),
            **image_disc_opt_config,
        )
        image_discriminator_scheduler = self.get_lr_scheduler(image_discriminator_optimizer, self.opt["image_discriminator"]["decay_epochs"])

        # report_disc_opt_config = {
        #     "lr" : self.opt["report_discriminator"]["learning_rate"],
        #     "betas" : (self.opt["report_discriminator"]["beta1"], self.opt["report_discriminator"]["beta2"])
        # }
        # report_discriminator_optimizer = self.optimizer(
        #     list(self.report_discriminator.parameters()),
        #     **report_disc_opt_config,
        # )
        # report_discriminator_scheduler = self.get_lr_scheduler(report_discriminator_optimizer, self.opt["report_discriminator"]["decay_epochs"])


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
    

    def report_adv_criterion(self, fake_report, real_report):
        # adversarial loss
        # additional discriminator network for reports (if needed)
        return self.report_adversarial_loss(fake_report, real_report)

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
    

    # def report_discriminator_step(self, valid, fake):
        # fake_report = self.buffer_reports(self.fake_report)
        ###########################################################################################
        # real_report_adv_loss = self.report_adv_criterion(self.report_discriminator(self.real_report), valid)
        # fake_report_adv_loss = self.report_adv_criterion(self.report_discriminator(fake_report.detach()), fake)
        ###########################################################################################

        # total_report_disc_loss = (real_report_adv_loss + fake_report_adv_loss) / 2
        # metrics = {
        #     "report_disc_loss": total_report_disc_loss,
        #     "report_disc_adv_loss_real": real_report_adv_loss,
        #     "report_disc_adv_loss_fake": fake_report_adv_loss,
        # }

        # self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        # return total_report_disc_loss


    def img_discriminator_step(self, valid, fake):
        # fake_report = self.buffer_reports(self.fake_report)
        fake_img = self.buffer_images(self.fake_img)
        # calculate loss for discriminator

        ###########################################################################################
        # calculate on real data
        real_img_adv_loss = self.img_adv_criterion(self.image_discriminator(self.real_img), valid)
        # calculate on fake data
        fake_img_adv_loss = self.img_adv_criterion(self.image_discriminator(fake_img.detach()), fake)
        ###########################################################################################

        # real_report_adv_loss = self.report_adv_criterion(self.report_discriminator(self.real_report), valid)
        # fake_report_adv_loss = self.report_adv_criterion(self.report_discriminator(fake_report.detach()), fake)

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
        self.real_report = batch['report']

        # gen_optimizer, disc_optimizer = self.optimizers()

        # generate valid and fake labels
        valid = Variable(
            Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape))),
            requires_grad=False
        )

        fake = Variable(
            Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape))),
            requires_grad=False
        )

        # generate fake reports and images
        self.fake_report = self.report_generator(self.real_img)
        self.fake_img = self.image_generator.sample(self.real_report)

        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator.sample(self.fake_report)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_loss = self.generator_step(valid)
            return gen_loss
        
        elif optimizer_idx == 2 or optimizer_idx == 3:
            img_disc_loss = self.discriminator_step(valid, fake)
            # report_disc_loss = self.report_discriminator_step(valid, fake)
            return img_disc_loss
        
        # # train generators
        # self.toggle_optimizer(gen_optimizer,0)
        # gen_loss = self.generator_step(valid)
        # gen_optimizer.zero_grad()
        # self.manual_backward(gen_loss)
        # gen_optimizer.step()
        # self.untoggle_optimizer(gen_optimizer)

        # # train discriminator
        # self.toggle_optimizer(disc_optimizer,1)
        # disc_loss = self.discriminator_step(valid, fake)
        # disc_optimizer.zero_grad()
        # self.manual_backward(disc_loss)
        # disc_optimizer.step()
        # self.untoggle_optimizer(disc_optimizer)


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
        num_samples = min(5, len(self.val_dataloader().dataset))
        val_samples = next(iter(self.val_dataloader()))

        # Generate reports and images for these samples
        generated_reports, generated_images = self(val_samples['target'])

        # Log the generated reports and images
        for i in range(num_samples):
            # Convert the tensor to a suitable image format (e.g., PIL Image)
            # and log using the logger (e.g., TensorBoard, WandB)
            # This is just an example, modify as per your logger and data format
            generated_image = generated_images[i].cpu().detach()
            # Convert image tensor to PIL Image or similar
            img_pil = transforms.ToPILImage()(generated_image.squeeze()).convert("RGB")
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format="JPEG")
            img_buffer.seek(0)
            # Log the image and the report text

            generated_report = generated_reports[i].cpu().detach()
            generated_report = torch.sigmpoid(generated_report)
            generated_report = (generated_report > 0.5).int()
            report_text = [self.opt['dataset']['chexpert_labels'][idx] for idx in range(len(generated_report)) if generated_report[idx] == 1]
            report_text = ', '.join(sorted(generated_report))

            self.logger.experiment.add_image(f"Generated Image {i}", img_buffer, self.current_epoch, dataformats='HWC')
            self.logger.experiment.add_text(f"Generated Report {i}", report_text, self.current_epoch)

    
    def _get_report_generator(self, model_name):
        if self.image_encoder_model == "Ark":
            return ARKModel(num_classes=self.num_classes,
                            ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK'])
        
        elif self.image_encoder_model == "BioVil":
            return BioViL(embedding_size=self.opt["report_generator"]["embedding_size"], 
                          num_classes=self.num_classes, 
                          hidden_1=self.opt["report_generator"]["classification_head_hidden1"],
                          hidden_2=self.opt["report_generator"]["classification_head_hidden2"], 
                          dropout_rate=self.opt["report_generator"]["dropout_prob"])
        else:
            raise NotImplementedError(f"Model {model_name} not implemented for report generation.")
        

    def _get_report_discriminator(self):
        # TODO : Implement additional discriminator network for reports (if needed)
        # and use classificationLoss as the adversarial criterion
        # return ReportDiscriminator(num_classes=self.num_classes)
        return ClassificationLoss(env_settings.MASTER_LIST[self.data_imputation])
    

    def _get_image_generator(self):
        # TODO : Implement image generator class
        return 
        
    def _get_image_discriminator(self):
        return ImageDiscriminator(channels=self.opt['image_discriminator']['channels'], 
                                  img_height=self.opt['image_discriminator']['img_height'],
                                  img_width=self.opt['image_discriminator']['img_width'])

    

        
        

