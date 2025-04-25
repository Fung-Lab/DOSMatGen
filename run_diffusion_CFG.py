from typing import List

import os
import wandb
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from spec2struct.dataset.datamodule import CrystalDataModule
from spec2struct.diffusion.diffusion_cfg import CSPDiffusion
from spec2struct.utils.utils import log_hyperparameters

def build_callbacks(config: DictConfig, save_dir) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in config.logging:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=config.logging.lr_monitor.logging_interval,
                log_momentum=config.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in config.train:
        callbacks.append(
            EarlyStopping(
                monitor=config.optim.lr_scheduler.monitor_metric,
                mode=config.optim.lr_scheduler.monitor_metric_mode,
                patience=config.train.early_stopping.patience,
                verbose=config.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in config.train:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                monitor=config.optim.lr_scheduler.monitor_metric,
                mode=config.optim.lr_scheduler.monitor_metric_mode,
                save_top_k=config.train.model_checkpoints.save_top_k,
                verbose=config.train.model_checkpoints.verbose,
                save_last=config.train.model_checkpoints.save_last,
            )
        )

    return callbacks

def run(config: DictConfig):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_dir = Path(f"{config.save_dir}/{timestamp}_{config.run_name}")
    os.makedirs(save_dir, exist_ok=True)

    # seed
    if config.train.deterministic:
        seed_everything(config.train.random_seed)
    
    # precision
    if config.train.trainer.precision == 32:
        torch.set_float32_matmul_precision('medium')

    # instantiate the data module
    data_module = CrystalDataModule(config)

    # instantiate model
    model = CSPDiffusion(**config)

    # instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(config, save_dir)

    # save scaler
    if data_module.scaler is not None:
        model.lattice_scaler = data_module.lattice_scaler.copy()
        model.scaler = data_module.scaler.copy()
    torch.save(data_module.lattice_scaler, save_dir / 'lattice_scaler.pt')
    torch.save(data_module.scaler, save_dir / 'prop_scaler.pt')

    # wandb logging
    wandb_logger = None
    if "wandb" in config.logging:
        wandb_config = config.logging.wandb
        wandb_logger = WandbLogger(
            name=config.run_name,
            group=config.run_name,
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.watch(
            model,
            log=config.logging.wandb_watch.log,
            log_freq=config.logging.wandb_watch.log_freq,
        )

    yaml_conf: str = OmegaConf.to_yaml(cfg=config)
    (save_dir / "hparams.yaml").write_text(yaml_conf)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=config.train.deterministic,
        check_val_every_n_epoch=config.logging.val_check_interval,
        **config.train.trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=config)

    trainer.fit(model=model, datamodule=data_module)
    # trainer.test(datamodule=data_module)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

conf = OmegaConf.load('configs/dos_cfg.yml')
print(OmegaConf.to_yaml(conf))

run(conf)