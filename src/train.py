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

    if params['trainer']['resume_training']:
        checkpoint = experiment + '/best_models/version_' + str(params['trainer']['resume_version']) + '/last.ckpt'
        trainer = pl.Trainer(accelerator="gpu",
                            max_epochs=params['trainer']['n_epoch'],
                            check_val_every_n_epoch=params['trainer']['check_val_every_n_epochs'],
                            log_every_n_steps=params["trainer"]["buffer_size"],
                            callbacks=[checkpoint_callback],
                            logger=logger,
                            resume_from_checkpoint = checkpoint)
        
    else:
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
    # torch.autograd.set_detect_anomaly(True)
    trainer.fit(CycleGAN_module, chexpert_data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_images', type=int, default=50000)
    parser.add_argument('--use_all_images', action='store_true', default=False)
    parser.add_argument('--soft_label_type', type=str, default=None)
    parser.add_argument('--report_gen_lr', type=float, default=0.0001)
    parser.add_argument('--without_decay', action='store_false', default=True)
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--resume_version', type=int, default=0)
    arguments = parser.parse_args()
    
    params = read_config(env_settings.CONFIG)
    params['dataset']['batch_size'] = arguments.batch_size
    params['dataset']['num_images'] = arguments.num_images
    params['dataset']['use_all_images'] = arguments.use_all_images
    params['trainer']['n_epoch'] = arguments.n_epoch
    params['trainer']['soft_label_type'] = arguments.soft_label_type
    params['trainer']['resume_training'] = arguments.resume_training
    params['trainer']['resume_version'] = arguments.resume_version
    params['report_generator']['learning_rate'] = arguments.report_gen_lr
    params['report_generator']['decay'] = arguments.without_decay
    
    main(params)
