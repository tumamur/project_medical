import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from modules.ImageComponentModule import ImageComponentModule
from modules.AdversarialClassificationModule import AdversarialClassificationModule, ReportBuffer
from modules.ChexpertModule import ChexpertDataModule
from tensorboard import program
from utils._prepare_data import DataHandler
from utils.environment_settings import env_settings
from utils.utils import read_config, get_monitor_metrics_mode


def start_tensorboard(port, tracking_address: str):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address, "--port", str(port)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    return tb


def main(params):
    torch.manual_seed(params["model"]["seed"])
    processor = DataHandler(opt=params["dataset"])
    chexpert_data_module = ChexpertDataModule(opt=params['dataset'], processor=processor)

    if params['model']['loss'] == 'Adversarial':
        ImageEncoder = AdversarialClassificationModule(opt=params)
    else:
        ImageEncoder = ImageComponentModule(opt=params)

    experiment = env_settings.EXPERIMENTS + '/' + params['model']['image_encoder_model']
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
        tb = start_tensorboard(env_settings.TENSORBOARD_PORT, experiment+"/lightning_logs") 
    except Exception as e:
        print(f"Could not start tensor board, got error {e}")

    # Start training
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(ImageEncoder, chexpert_data_module)


if __name__ == '__main__':
    params = read_config(env_settings.CONFIG)
    main(params)
