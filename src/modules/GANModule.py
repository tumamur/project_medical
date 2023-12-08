import torch as nn
import torch.nn as nn
from losses.CombinationLoss import CombinationLoss
from losses.Test_loss import ClassificationLoss
from losses.Perceptual_loss import PerceptualLoss
from models.Ark import ArkModel
from models.BioViL import BioViL
from utils.environment_settings import env_settings
from utils.utils import read_config
from torchvision.utils import save_image


class CycleGAN(nn.Module):
    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        # Define Generators
        if opt['model']['image_encoder_model'] == 'Ark':
            self.gen_class_to_image = ArkModel(opt['model']['num_classes'], env_settings.PRETRAINED_PATH['Ark'])
        elif opt['model']['image_encoder_model'] == 'BioVil':
            self.gen_image_to_class = BioViL(opt['model']['embedding_size'], opt['model']['num_classes'],
                                             opt['model']['classification_head_hidden1'],
                                             opt['model']['classification_head_hidden2'], opt['model']['dropout_prob'])
        self.gen_class_to_image = 'test'

        # Define Discriminators
        self.image_disc = 'test'
        self.class_disc = ClassificationLoss(env_settings.MASTER_LIST['zeros'])

        # Define Loss Functions
        self.cycle_class_loss = nn.BCEWithLogitsLoss()
        self.cycle_image_loss = PerceptualLoss()

        # Define tracking metrics
        self.Class_metric = CombinationLoss(opt['model']['loss'], opt['dataset']["data_imputation"], opt['dataset']['chexpert_labels'], opt['model']["threshold"])

    def forward(self, real_class, real_image):
        # Implement the forward pass of the CycleGAN
        fake_class = self.gen_image_to_class(real_image)
        fake_image = self.gen_image_to_class(real_class)

        # Implement forward and backward cycle
        cycle_image = self.gen_image_to_class(fake_class)
        cycle_class = self.gen_image_to_class(fake_image)

        return fake_class, fake_image, cycle_class, cycle_image

    def train_step(self, batch, batch_idx):
        GAN_cycle_losses = cycle_gan.cycle_image_loss(real_image, fake_image) + cycle_gan.cycle_class_loss(real_class,
                                                                                                           fake_class)
        GAN_adversarial_losses = cycle_gan.class_disc(fake_class)

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
    cycle_gan = CycleGAN(params)


if __name__ == '__main__':
    main()










