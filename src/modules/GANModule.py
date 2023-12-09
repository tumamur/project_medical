import torch
import torch as nn
import torch.nn as nn
from tensorboard import program
from torchvision.utils import save_image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from losses.CombinationLoss import CombinationLoss
from losses.Test_loss import ClassificationLoss
from losses.Perceptual_loss import PerceptualLoss
from models.Ark import ArkModel
from models.BioViL import BioViL
from utils.environment_settings import env_settings
from modules.Image_discriminator import ImageDiscriminator
from utils._prepare_data import DataHandler
from modules.ChexpertModule import ChexpertDataModule
from utils.utils import read_config, get_monitor_metrics_mode
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def start_tensorboard(port, tracking_address: str):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address, "--port", str(port)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    return tb

class CycleGAN(pl.LightningModule):
    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.opt = opt
        # Define Generators
        if opt['model']['image_encoder_model'] == 'Ark':
            self.gen_class_to_image = ArkModel(opt['model']['num_classes'], env_settings.PRETRAINED_PATH['Ark'])
        elif opt['model']['image_encoder_model'] == 'BioVil':
            self.gen_image_to_class = BioViL(opt['model']['embedding_size'], opt['model']['num_classes'],
                                             opt['model']['classification_head_hidden1'],
                                             opt['model']['classification_head_hidden2'], opt['model']['dropout_prob'])
        self.gen_class_to_image = 'test'

        # Define Discriminators
        self.image_disc = ImageDiscriminator(channels=opt['model']['channels'], img_height=opt['model']['img_height'],
                                             img_width=opt['model']['img_width'])
        self.class_disc = ClassificationLoss(env_settings.MASTER_LIST['zeros'])

        # Define Loss Functions
        self.cycle_class_loss = nn.BCEWithLogitsLoss()
        self.cycle_image_loss = PerceptualLoss()
        self.adversarial_criterion = nn.MSELoss()

        # Define tracking metrics
        self.Class_metric = CombinationLoss(opt['model']['loss'], opt['dataset']["data_imputation"],
                                            opt['dataset']['chexpert_labels'], opt['model']["threshold"])

        # Define lambda
        self.lambda_cycle_loss = opt['model']['lambda_cycle_loss']

    def forward(self, real_class, real_image):
        # Implement the forward pass of the CycleGAN
        fake_class = self.gen_image_to_class(real_image)
        fake_image = self.gen_class_to_image(real_class)

        # Implement forward and backward cycle
        cycle_image = self.gen_class_to_image(fake_class)
        cycle_class = self.gen_image_to_class(fake_image)

        return fake_class, fake_image, cycle_class, cycle_image

    def configure_optimizers(self):
        generator_params = list(self.gen_class_to_image.parameters()) + list(self.gen_image_to_class.parameters())
        discriminator_params = list(self.image_disc.parameters()) + list(self.class_disc.parameters())
        optimizer_generator = optim.Adam(
            generator_params,
            lr=self.opt['generator']['learning_rate'],
            betas=(self.opt['generator']['beta1'], self.opt['generator']['beta2'])
        )

        optimizer_discriminator = optim.Adam(
            discriminator_params,
            lr=self.opt['discriminator']['learning_rate'],
            betas=(self.opt['discriminator']['beta1'], self.opt['discriminator']['beta2'])
        )
        return {
            'generator_optimizer': optimizer_generator,
            'discriminator_optimizer': optimizer_discriminator,
            # Optionally, you can include a learning rate scheduler
            'scheduler': {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer_generator, step_size=10, gamma=0.9),
                'monitor': self.opt['discriminator']['monitor'],  # Adjust this based on your monitoring metric
            }
        }

    def train_step(self, batch, batch_idx, optimizer_idx):
        real_class, real_image = batch

        # Creating ground truths for adversarial loss
        valid = Variable(
            Tensor(np.ones((real_image.size(0), *self.image_disc.output_shape))),
            requires_grad=False
        )
        fake = Variable(
            Tensor(np.ones((real_image.size(0), *self.image_disc.output_shape))),
            requires_grad=False
        )

        if optimizer_idx == 0:
            generator_loss = self.generator_step(real_image, real_class, valid)

            return generator_loss

        elif optimizer_idx == 1:
            discriminator_loss = self.discriminator_step(real_image, real_class, valid, fake)

            return discriminator_loss

    def generator_step(self, real_image, real_class, valid):
        # Forward pass
        fake_class, fake_image, cycle_class, cycle_image = self.forward(real_class, real_image)

        # Compute GAN cycle losses
        cycle_loss_image = self.cycle_image_loss(real_image, cycle_image)
        cycle_loss_class = self.cycle_class_loss(real_class, cycle_class)
        cycle_loss = (cycle_loss_image + cycle_loss_class) / 2

        # Adversarial loss
        adv_loss_class = self.class_disc(fake_class)
        adv_loss_image = self.adversarial_criterion(self.image_disc(self.image_disc(fake_image), valid))
        adv_loss = (adv_loss_image + adv_loss_class) / 2

        # Total generator loss
        generator_loss = self.lambda_cycle_loss * cycle_loss + adv_loss

        # Log losses
        self.log('cycle_loss_image', cycle_loss_image, prog_bar=True)
        self.log('cycle_loss_class', cycle_loss_class, prog_bar=True)
        self.log('cycle_loss', cycle_loss, prog_bar=True)
        self.log('adv_loss_class', adv_loss_class, prog_bar=True)
        self.log('adv_loss_image', adv_loss_image, prog_bar=True)
        self.log('adv_loss', adv_loss, prog_bar=True)
        self.log('generator_loss', generator_loss, prog_bar=True)

        return generator_loss

    def discriminator_step(self, real_image, real_class, valid, fake):
        # Forward pass
        fake_class, fake_image, cycle_class, cycle_image = self.forward(real_class, real_image)

        # Discriminator loss
        adv_loss_image_real = self.adversarial_criterion(self.image_disc(real_image), valid)
        adv_loss_image_fake = self.adversarial_criterion(self.image_disc(fake_image), fake)

        # Total discriminator loss
        discriminator_loss = (adv_loss_image_real + adv_loss_image_fake) / 2

        # Log losses (optional)
        self.log('real_adversarial_loss', adv_loss_image_real, prog_bar=True)
        self.log('fake_adversarial_loss', adv_loss_image_fake, prog_bar=True)
        self.log('discriminator_loss', discriminator_loss, prog_bar=True)

        return discriminator_loss


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


def save_image_samples(batches_done, cycle_gan, val_dataloader):
    print("batches_done", batches_done)

    imgs = next(iter(val_dataloader))

    cycle_gan.gen_image_to_class.eval()
    cycle_gan.gen_class_to_image.eval()

    # Arange images along x-axis
    real_images = make_grid(real_images, nrow=8, normalize=True)
    fake_images = make_grid(fake_images, nrow=8, normalize=True)
    real_classes = make_grid(real_classes, nrow=8, normalize=True)
    fake_classes = make_grid(fake_classes, nrow=8, normalize=True)



def show_image_samples(batches_done):
    pass

def main():
    params = read_config(env_settings.CONFIG)
    torch.manual_seed(params["model"]["seed"])
    processor = DataHandler(opt=params["dataset"])
    chexpert_data_module = ChexpertDataModule(opt=params['dataset'], processor=processor)
    cycle_gan = CycleGAN(params)

    experiment = env_settings.EXPERIMENTS + '/CycleGAN/' + params['model']['image_encoder_model']
    logger = TensorBoardLogger(experiment, default_hp_metric=False)

    monitor = params['model']['monitor']
    try:
        monitor_metrics_mode = get_monitor_metrics_mode()[monitor]
    except:
        raise Exception(f"Monitor {monitor} not found in the config file")

    file_name = params['model']['image_encoder_model'] + "_{epoch:02d}_{" + monitor + ":.4f}"

    checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=f'{experiment}/best_models/version_{logger.version}',
            filename=file_name,
            save_top_k=1,
            mode=monitor_metrics_mode,
    )

    # Create trainer
    trainer = pl.Trainer(accelerator="gpu",
                        max_epochs=params['model']['n_epoch'],
                        check_val_every_n_epoch=params['model']['check_val_every_n_epochs'],
                        log_every_n_steps=params['model']['accumulated_batches']*params['dataset']['batch_size'],
                        callbacks=[checkpoint_callback],
                        logger=logger
                        )

    # start tensorboard
    try:
        tb = start_tensorboard(env_settings.TENSORBOARD_PORT, experiment + "/lightning_logs")
    except Exception as e:
        print(f"Could not start tensor board, got error {e}")

    # Start training
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(cycle_gan, chexpert_data_module)


if __name__ == '__main__':
    main()



