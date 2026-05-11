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
import argparse
import dataclasses
from pathlib import Path

import pytorch_lightning as pl
import torch

from config_utils import ExperimentConfig, load_config
from datasets import load_dataset
from models import INR
from utils import _log_dataset_props, initialise_logger, run_trainer

torch.set_float32_matmul_precision("medium")


def prepare_argparser():
    parser = argparse.ArgumentParser(description="Train an INR.")
    parser.add_argument("--conf", default=None, type=str)
    parser.add_argument("--config_id", default=0, type=int)
    parser.add_argument(
        "--dataset_sources",
        default=None,
        type=str,
        help="List of folders with images.",
    )
    parser.add_argument("--datasets", default=None, type=str, help="List of files.")
    parser.add_argument("--count", action="store_true")
    return parser


def train_from_config(
    experiment_config: ExperimentConfig,
):
    pl.seed_everything(experiment_config.seed)
    dataset = load_dataset(experiment_config)

    model = INR(
        dataset=dataset,
        experiment_config=experiment_config,
    )

    wandb_logger = initialise_logger(experiment_config)
    _log_dataset_props(wandb_logger, dataset)
    for name, value in experiment_config.logger_config.extra_config.items():
        wandb_logger.experiment.config[name] = value
    wandb_logger.experiment.config["config_hash"] = experiment_config.config_hash

    run_trainer(
        wandb_logger,
        model,
        dataset,
        experiment_config.fast_dev_run,
        accelerator=experiment_config.accelerator,
        checkpoints_path=experiment_config.checkpoints_path,
        checkpoints_frequency=experiment_config.checkpoints_frequency,
        max_steps=experiment_config.epochs,
        val_check_interval=experiment_config.logger_config.val_check_interval,
    )


def prep_datasets(opt: argparse.Namespace):
    if opt.dataset_sources is not None:
        dataset_sources = eval(opt.dataset_sources)
        datasets = []
        for dataset_source in dataset_sources:
            datasets.extend(
                [
                    str(x.absolute())
                    for x in Path(dataset_source).iterdir()
                    if x.is_file()
                ]
            )
    elif opt.datasets is not None:
        datasets = eval(opt.datasets)
    else:
        raise ValueError("Either --dataset_sources or --datasets should be provided")

    return sorted(datasets)


if __name__ == "__main__":
    parser = prepare_argparser()
    opt, extras = parser.parse_known_args()
    experiment_config = load_config(opt.conf, cli_args=extras)
    datasets = prep_datasets(opt)

    run_configs = experiment_config.prepare_run_configs(datasets=datasets)
    if opt.count:
        print(f"Total number of configurations: {len(run_configs)}")
        exit(0)

    run_config = run_configs[opt.config_id]
    extra_config = run_config.logger_config.extra_config.copy()
    extra_config["config_id"] = opt.config_id
    logger_config = dataclasses.replace(
        run_config.logger_config, extra_config=extra_config
    )
    run_config = dataclasses.replace(run_config, logger_config=logger_config)
    train_from_config(run_config)
