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
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(CycleGAN_module, chexpert_data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--update_freq', type=int, default=1000)
    parser.add_argument('--use_float_reports', action='store_true', default=False)
    parser.add_argument('--adaptive_threshold_disc', action='store_true', default=False)
    parser.add_argument('--adaptive_threshold_gen', action='store_true', default=False)
    parser.add_argument('--resume_training', action='store_true', default=False)
    
    parser.add_argument('--resume_version', type=int, default=2)
    parser.add_argument('--img_gen_lr', type=float, default=0.0001)
    parser.add_argument('--img_gen_beta', type=float, default=0.5)
    parser.add_argument('--report_gen_lr', type=float, default=0.0001)
    parser.add_argument('--report_gen_beta', type=float, default=0.5)
    parser.add_argument('--img_disc_lr', type=float, default=0.0001)
    parser.add_argument('--img_disc_beta', type=float, default=0.5)
    parser.add_argument('--report_disc_lr', type=float, default=0.0001)
    parser.add_argument('--report_disc_beta', type=float, default=0.5)

    parser.add_argument('--lambda_cycle_loss', type=int, default=10)
    
    arguments = parser.parse_args()

    
    params = read_config(env_settings.CONFIG)
    params['trainer']['n_epoch'] = arguments.n_epochs
    params['dataset']['batch_size'] = arguments.batch_size
    params['trainer']['update_freq'] = arguments.update_freq
    params['trainer']['use_float_reports'] = arguments.use_float_reports
    params['trainer']['adaptive_threshold_gen'] = arguments.adaptive_threshold_gen
    params['trainer']['lambda_cycle_loss'] = arguments.lambda_cycle_loss
    params['image_generator']['learning_rate'] = arguments.img_gen_lr
    params['image_discriminator']['learning_rate'] = arguments.img_disc_lr
    params['report_generator']['learning_rate'] = arguments.report_gen_lr
    params['report_discriminator']['learning_rate'] = arguments.report_disc_lr
    params['image_generator']['beta'] = arguments.img_gen_beta
    params['image_discriminator']['beta'] = arguments.img_disc_beta
    params['report_generator']['beta'] = arguments.report_gen_beta
    params['report_discriminator']['beta'] = arguments.report_disc_beta
    params['trainer']['resume_training'] = arguments.resume_training
    params['trainer']['resume_version'] = arguments.resume_version
    main(params)
