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
import deepspeed as ds

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
        self.betas = opt['model']["betas"]
        self.optimizer_name = opt['model']["optimizer"]
        self.scheduler_name = opt['model']["scheduler"]
        self.weight_decay = opt['model']["weight_decay"]
        self.decay_epochs = opt['model']["n_epoch"]
        self.n_epoch = opt['model']["n_epoch"]

        self.image_encoder_model = opt['model']["image_encoder_model"]
        self.hidden_dim1 = opt['model']["classification_head_hidden1"]
        self.hidden_dim2 = opt['model']["classification_head_hidden2"]
        self.dropout_rate = opt['model']["dropout_prob"]
        self.buffer_size = opt['model']["buffer_size"]

        self.generator = self._get_model()
        self.discriminator = self._get_discriminator()
        self.buffer_fake_labels = ReportBuffer(self.buffer_size)

        optimizer_dict = {
            'Adam': ds.ops.adam.FusedAdam, 
            'AdamW': torch.optim.AdamW,
        }

        self.optimizer = optimizer_dict[self.optimizer_name]

    def forward(self, real_image):
        return self.generator(real_image)
    
    def init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.generator.apply(init_fn)
        self.discriminator.apply(init_fn)


    def setup(self, stage):
        if stage == 'fit':
            self.init_weights()
            print("Model initialized")
    
    def get_lr_scheduler(self, optimizer):
        def lr_lambda(epoch):
            len_decay_phase = self.n_epochs - self.decay_epochs + 1.0
            curr_decay_step = max(0, epoch - self.decay_epochs + 1.0)
            val = 1.0 - curr_decay_step / len_decay_phase
            return max(0.0, val)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        opt_config = {
            "lr": self.learning_rate,
            "betas": self.betas,
        }
        opt_gen = self.optimizer(
            list(self.generator.parameters()),
            **opt_config,
        )
        opt_disc = self.optimizer(
            list(self.discriminator.parameters()),
            **opt_config,
        )
        optimizers = [opt_gen, opt_disc]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers

    def adversarial_criterion(self, y_hat, y):
        return nn.functional.mse_loss()(y_hat, y)

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
        self.fake_labels = self.generator(self.real_image)
        self.fake_labels = torch.sigmoid(self.fake_labels)
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

    def on_train_epoch_start(self):
        # record learning rates
        curr_lr = self.lr_schedulers()[0].get_last_lr()[0]
        self.log("lr", curr_lr, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        # update learning rates
        for sch in self.lr_schedulers():
            sch.step()
        
        # print current state of epoch
        logged_values = self.trainer.progress_bar_metrics
        print(
            f"Epoch {self.current_epoch+1}",
            *[f"{k}: {v:.5f}" for k, v in logged_values.items()],
            sep=" - ",
        )

    def on_train_end(self):
        print("Training ended.")
    
    def _get_model(self):
        if self.image_encoder_model == "Ark":
            return ARKModel(num_classes=self.num_classes, ark_pretrained_path=env_settings.PRETRAINED_PATH['ARK'])
        elif self.image_encoder_model == "BioVil":
            return BioViL(embedding_size=self.embedding_size, num_classes=self.num_classes, hidden_1=self.hidden_dim1,
                          hidden_2=self.hidden_dim2, dropout_rate=self.dropout_rate)

    def _get_discriminator(self):
        return ReportDiscriminator(input_dim=self.num_classes, output_dim=self.num_classes)
