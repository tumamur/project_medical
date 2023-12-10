import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from models.BioViL import BioViL
from models.ARK import ARKModel
from models.Discriminator import ReportDiscriminator
from losses.Metrics import Metrics
from torch.optim import lr_scheduler
from utils.environment_settings import env_settings
import torchmetrics.functional as F
from losses.ClassificationLoss import SimilarityLoss, AdversarialLoss
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
from torch.autograd import Variable
import numpy as np

torch.autograd.set_detect_anomaly(True)

class ReportBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            # the current capacity of the buffer
            self.curr_cap = 0
            # initialize buffer as empty list
            self.buffer = []


    def __call__(self, fake_labels):
        # the buffer is not used
        if self.buffer_size == 0:
            return fake_labels
        
        return_labels = []
        for label in fake_labels:
            if self.curr_cap < self.buffer_size:
                self.curr_cap += 1
                self.buffer.append(label)
                return_labels.append(label)
            else:
                p = np.random.uniform(0, 1)
                # swap the buffer with probability 0.5
                if p > 0.5:
                    idx = np.random.randint(0, self.buffer_size)
                    return_labels.append(self.buffer[idx].clone())
                    self.buffer[idx] = label
                else:
                    return_labels.append(label)

        return torch.stack(return_labels)


class AdversarialClassificationModule(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.automatic_optimization = False

        self.num_classes = opt['model']["num_classes"]
        self.embedding_size = opt['model']["embedding_size"]
        self.learning_rate = opt['model']["learning_rate"]
        self.optimizer_name = opt['model']["optimizer"]
        self.scheduler_name = opt['model']["scheduler"]
        self.weight_decay = opt['model']["weight_decay"]
        self.n_epoch = opt['model']["n_epoch"]

        self.image_encoder_model = opt['model']["image_encoder_model"]
        self.hidden_dim1 = opt['model']["classification_head_hidden1"]
        self.hidden_dim2 = opt['model']["classification_head_hidden2"]
        self.dropout_rate = opt['model']["dropout_prob"]
        self.buffer_size = opt['model']["buffer_size"]

        self.generator = self._get_model()
        self.discriminator = self._get_discriminator()
        self.buffer_fake_labels = ReportBuffer(self.buffer_size)

    def forward(self, real_image):
        real_image = real_image.float()
        fake_class = self.generator(real_image)
        return fake_class
    
    def get_lr_scheduler(self, optimizer):
        scheduler_dict = {
            "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch, eta_min=1e-7),
            "exponential": lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95),
            "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                                verbose=True),
        }
        if self.scheduler_name in scheduler_dict:
            return scheduler_dict[self.scheduler_name]
        return None

    def configure_optimizers(self):
        optimizer_dict = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            'D_Adam' : torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            'D_AdamW' : torch.optim.AdamW(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
        }

        gen_optimizer = optimizer_dict[self.optimizer_name]
        d_optimizer_name = self.optimizer_name.replace('Adam', 'D_Adam')
        disc_optimizer = optimizer_dict[d_optimizer_name]
        optimizers = [gen_optimizer, disc_optimizer]
        #schedulers = [self.get_lr_scheduler(optimizer) for optimizer in optimizers]
        #return optimizers, schedulers
        return optimizers

    def adversarial_criterion(self, y_hat, y):
        return nn.BCEWithLogitsLoss()(y_hat, y)
    
    def get_adv_loss(self, fake, disc):
        fake_hat = disc(fake)
        real_labels = torch.ones_like(fake_hat)
        adv_loss = self.adversarial_criterion(fake_hat, real_labels)
        return adv_loss

    def get_gen_loss(self):
        # adversarial loss
        adversarial_loss = self.get_adv_loss(self.fake_labels, self.discriminator)
        return adversarial_loss

    def get_disc_loss(self, real, fake, disc):
        # calculate loss on real data
        real_hat = disc(real)
        real_labels = torch.ones_like(real_hat)
        real_loss = self.adversarial_criterion(real_hat, real_labels)

        # calculate loss on fake data
        fake_hat = disc(fake.detach())
        fake_labels = torch.zeros_like(fake_hat)
        fake_loss = self.adversarial_criterion(fake_hat, fake_labels)

        # combine losses
        disc_loss = (real_loss + fake_loss) / 2
        return disc_loss
    
    def get_disc_loss_on_buffer(self):
        fake_labels = self.buffer_fake_labels(self.fake_labels)
        return self.get_disc_loss(self.real_labels, fake_labels, self.discriminator)


    def training_step(self, train_batch, batch_idx):
        self.real_image = train_batch['target']
        self.real_labels = train_batch['report']

        opt_gen, opt_disc = self.optimizers()

        # generate fake labels
        self.fake_labels = self.forward(self.real_image)
        #  print(self.fake_labels)
        # train generators
        self.toggle_optimizer(opt_gen, 0)
        gen_loss = self.get_gen_loss()
        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)

        # train discriminator
        self.toggle_optimizer(opt_disc, 1)
        disc_loss = self.get_disc_loss_on_buffer()
        opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        # record logs
        metrics = {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)


    def validation_step(self, val_batch, batch_idx):
        pass

    def _get_model(self):
        if self.image_encoder_model == "Ark":
            return ARKModel(num_classes=self.num_classes, ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK'])
        elif self.image_encoder_model == "BioVil":
            return BioViL(embedding_size=self.embedding_size, num_classes=self.num_classes, hidden_1=self.hidden_dim1,
                          hidden_2=self.hidden_dim2, dropout_rate=self.dropout_rate)

    def _get_discriminator(self):
        return ReportDiscriminator(input_dim=self.num_classes, output_dim=self.num_classes)
