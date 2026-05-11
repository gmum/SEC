"""
Copyright (C) 2025 Adam Kania, Tomasz Dądela, Maciej Rut, Przemysław Spurek, GlaxoSmithKline plc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
import torch.utils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger

from datasets import DummyValidationDataset


def _log_dataset_props(wandb_logger, dataset):
    wandb_logger.experiment.config["dataset"] = dataset.name
    wandb_logger.experiment.config["height"] = dataset.resolution[0]
    wandb_logger.experiment.config["width"] = dataset.resolution[1]
    wandb_logger.experiment.config["total_pixels"] = (
        dataset.resolution[0] * dataset.resolution[1]
    )
    wandb_logger.experiment.config["channels"] = dataset.channels


def run_trainer(
    wandb_logger,
    model,
    dataset,
    fast_dev_run,
    accelerator,
    checkpoints_path,
    max_steps,
    val_check_interval,
    checkpoints_frequency,
):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        DummyValidationDataset(), batch_size=1, num_workers=0
    )

    timer = Timer(duration=None, verbose=False)

    trainer = Trainer(
        max_epochs=-1,
        max_steps=max_steps,
        accelerator=accelerator,
        logger=wandb_logger,
        devices=1,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=1,
        fast_dev_run=fast_dev_run,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[timer],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Redo validation step, but save high-resolution images
    # trainer.test(model=model, dataloaders=val_loader)

    for name in ["train", "validate", "test"]:
        print(f"{name}: {timer.time_elapsed(name) / 60:.1f} min")
    wandb_logger.experiment.finish()


def initialise_logger(experiment_config):
    model_config = experiment_config.model_config
    logger_config = experiment_config.logger_config

    name = f"{model_config.config_name}-seed={experiment_config.seed}-dataset={experiment_config.dataset_id}"
    if logger_config.debug_id:
        name += f"-debug_id={logger_config.debug_id}"

    config = experiment_config.dict()
    config.update(model_config.dict())
    wandb_logger = WandbLogger(
        project=logger_config.wandb_project,
        group=logger_config.group,
        name=name,
        save_dir=logger_config.save_dir,
        # dir=logger_config.save_dir,
        config=config,
    )
    return wandb_logger
