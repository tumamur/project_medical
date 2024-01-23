import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from modules.diffusion import ContextUnet, DDPM
from utils.utils import read_config

def main(params):
    device = "cuda:0"
    pl.seed_everything(params['trainer_diffusion']['seed'])
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((params['image_generator_diffusion']['image_size'], params['image_generator_diffusion']['image_size'])), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = ImageFolder('../data/xray_train', transform=tf)
    dataset_val = ImageFolder('../data/xray_val', transform=tf)

    dataloader_train = DataLoader(dataset_train, batch_size=params['trainer_diffusion']['batch_size'], shuffle=True, num_workers=5)
    dataloader_val = DataLoader(dataset_val, batch_size=300, shuffle=False, num_workers=5)

    logger = TensorBoardLogger('tb_logs', name='my_model')

    model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=params['image_generator_diffusion']['n_feat'], n_classes=params['trainer_diffusion']['num_classes']), 
                 betas=(1e-4, 0.02), n_T=params['image_generator_diffusion']['n_T'],
                 device=device, drop_prob=0.1,
                 epochs=params['trainer_diffusion']['n_epoch'], image_size=params['image_generator_diffusion']['image_size'],
                 n_classes=params['trainer_diffusion']['num_classes'])
    trainer = pl.Trainer(gpus=1, max_epochs=params['trainer_diffusion']['n_epoch'], num_sanity_val_steps=0, logger=logger)
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == '__main__':
    params = read_config('./config.yaml')
    main(params)
