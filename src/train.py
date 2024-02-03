import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from modules.ChexpertModule import ChexpertDataModule
from modules.CycleGANModule import CycleGAN
from tensorboard import program
from utils._prepare_data import DataHandler
from utils.environment_settings import env_settings
from utils.utils import read_config


def start_tensorboard(port, tracking_address: str):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address, "--port", str(port)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    return tb


def main(params):
    torch.manual_seed(params["trainer"]["seed"])
    processor = DataHandler(opt=params["dataset"])
    chexpert_data_module = ChexpertDataModule(opt=params['dataset'], processor=processor)
    CycleGAN_module = CycleGAN(opt=params)
    experiment = (env_settings.EXPERIMENTS + params['image_generator']['model']
                  + "_" + params["report_generator"]["model"])


    logger = TensorBoardLogger(experiment, default_hp_metric=False)

    if params['trainer']['save_images']:
        save_images_path = f'{experiment}/images/version_{logger.version}'
        os.makedirs(save_images_path, exist_ok=True)
        params['trainer']['save_images_path'] = save_images_path
        print(f"Images will be saved to {save_images_path}")

    file_name = "cycle_gan" + "_{epoch:02d}"
    checkpoint_callback = ModelCheckpoint(
            dirpath=f'{experiment}/best_models/version_{logger.version}',
            filename=file_name,
            save_last=1,
    )

    # Create trainer
    trainer = pl.Trainer(accelerator="gpu",
                        max_epochs=params['trainer']['n_epoch'],
                        check_val_every_n_epoch=params['trainer']['check_val_every_n_epochs'],
                        log_every_n_steps=params["trainer"]["buffer_size"],
                        callbacks=[checkpoint_callback],
                        logger=logger)

    # start tensorboard
    try:
        tb = start_tensorboard(env_settings.TENSORBOARD_PORT, experiment+"/lightning_logs") 
    except Exception as e:
        print(f"Could not start tensor board, got error {e}")

    # Start training
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(CycleGAN_module, chexpert_data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--image_generator', default='cgan')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--dataset_size', type=int, default=-1) # -1 : all
    arguments = parser.parse_args()
    
    params = read_config(env_settings.CONFIG)
    params['dataset']['sample_size'] = arguments.dataset_size
    params['image_generator']['model'] = arguments.image_generator
    params['trainer']['n_epoch'] = arguments.n_epochs
    main(params)
