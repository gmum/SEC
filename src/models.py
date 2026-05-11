"""
Copyright (C) 2025 Adam Kania, Tomasz Dądela, Maciej Rut, Przemysław Spurek, GlaxoSmithKline plc

Code based on:
- https://github.com/liuzhen0212/FINER
- https://github.com/vsitzmann/siren

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
from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import rff.layers
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch import nn

from config_utils import (
    ExperimentConfig,
    FinerConfig,
    MLPConfig,
    ReLUConfig,
    SirenConfig,
    WireConfig,
)
from data_utils import tensor_to_image
from datasets import Signal
from wire import Wire


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega: list[float] | float = 30.0,
        beta=1.0,
        weight_constant=6.0,
        is_linear=False,
        apply_omega_when_linear=False,
        boost_bias=False,
    ):
        super().__init__()
        if is_linear and not apply_omega_when_linear:
            # Don't apply gradient boosting on the last layer
            # (not used; we follow the original SIREN implementation)
            omega = 1.0
        if isinstance(omega, list) and in_features != len(omega):
            raise ValueError("Omega must be provided for each input feature.")
        if isinstance(omega, list) and not is_first:
            raise ValueError("Omega can only be a list for the first layer.")

        self.omega = omega
        self.is_first = is_first
        self.boost_bias = boost_bias
        self.beta = beta
        self.weight_constant = weight_constant
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = lambda x: x
        if not is_linear:
            self.activation = torch.sin

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                num_input = self.linear.weight.size(-1)
                w = torch.distributions.beta.Beta(self.beta, self.beta).rsample(
                    self.linear.weight.shape
                )
                w = w * 2 - 1
                w = w / num_input
                self.linear.weight.copy_(w)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(self.weight_constant / self.in_features) / self.omega,
                    np.sqrt(self.weight_constant / self.in_features) / self.omega,
                )

    def forward(self, x):
        if not self.boost_bias:
            if isinstance(self.omega, list):
                x = torch.clone(x)
                for i, omega in self.omega:
                    x[:, i] = omega * x[:, i]
            else:
                x = self.omega * x
            return self.activation(self.linear(x))
        else:
            return self.activation(self.omega * self.linear(x))


class BaseMLP(ABC, nn.Module):
    @property
    @abstractmethod
    def net(self) -> nn.Sequential:
        pass

    @net.setter
    def net(self, net):
        self.net = net

    @property
    @abstractmethod
    def spectral_parameters(self) -> tuple:
        pass

    @spectral_parameters.setter
    def spectral_parameters(self, spectral_parameters):
        self.spectral_parameters = spectral_parameters

    def forward(self, coords):
        return self.net(coords)

    def forward_in_batches(self, coords, batch_size=2**19) -> torch.Tensor:
        return inference_in_batches(
            self.net,
            coords,
            batch_size=batch_size,
            device=self.net.parameters().__next__().device,
        )


class Siren(BaseMLP):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega=30.0,
        beta=1.0,
        weight_constant=6.0,
        apply_omega_when_linear=False,
        boost_bias=False,
    ):
        super().__init__()

        self._spectral_parameters = (first_omega_0, beta, weight_constant)
        self._net = []
        self._net.append(
            SineLayer(
                in_features,
                hidden_features,
                beta=beta,
                is_first=True,
                omega=first_omega_0,
            )
        )

        for i in range(hidden_layers):
            self._net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega=hidden_omega,
                    weight_constant=weight_constant,
                    boost_bias=boost_bias,
                )
            )
        self._net.append(
            SineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega=hidden_omega,
                is_linear=outermost_linear,
                apply_omega_when_linear=apply_omega_when_linear,
                boost_bias=boost_bias,
            )
        )

        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


class FinerLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30.0,
        first_bias_scale=None,
        scale_req_grad=False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out


class Finer(BaseMLP):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=30,
        hidden_omega_0=30.0,
        bias=True,
        first_bias_scale=None,
        scale_req_grad=False,
    ):
        self._spectral_parameters = (first_omega_0, first_bias_scale)
        super().__init__()
        self._net = []
        self._net.append(
            FinerLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad,
            )
        )

        for i in range(hidden_layers):
            self._net.append(
                FinerLayer(
                    hidden_features,
                    hidden_features,
                    omega_0=hidden_omega_0,
                    scale_req_grad=scale_req_grad,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )
        self._net.append(final_linear)
        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


class FourierFeaturesMLP(BaseMLP):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        sigma=1.0,
    ):
        super().__init__()

        self._spectral_parameters = (sigma,)
        self._net = []
        self._net.append(
            rff.layers.GaussianEncoding(
                sigma=sigma, input_size=in_features, encoded_size=hidden_features // 2
            )
        )

        for i in range(hidden_layers):
            self._net.append(torch.nn.Linear(hidden_features, hidden_features))
            self._net.append(torch.nn.ReLU())
        self._net.append(torch.nn.Linear(hidden_features, out_features))

        self._net = nn.Sequential(*self._net)

    @property
    def net(self) -> nn.Sequential:
        return self._net

    @property
    def spectral_parameters(self) -> tuple:
        return self._spectral_parameters


class INRBase(pl.LightningModule):
    def __init__(
        self,
        dataset,
        experiment_config: ExperimentConfig,
        gt_example_image: torch.Tensor | None = None,
    ):
        super().__init__()
        self.experiment_config = experiment_config
        self.dataset = dataset

        self.net = init_net_from_config(experiment_config.model_config, dataset)

        self.gt_spectra_normalised = None
        self.gt_example_image = gt_example_image

    @property
    def wandb_logger(self) -> WandbLogger:
        return self.logger

    def _log_summary_table(self, metrics: dict):
        table_keys = sorted(metrics.keys())
        table_data = [metrics[key] for key in metrics]
        self.wandb_logger.log_metrics(metrics, step=self.global_step)

        self.wandb_logger.log_table(
            key="summary_table",
            columns=table_keys,
            data=[table_data],
            step=self.global_step,
        )

    def validation_step(self, _):
        # everything is done in on_validation_epoch_end
        pass

    def test_step(self, _):
        # everything is done in on_test_epoch_end
        pass


class INR(INRBase):
    def __init__(
        self,
        dataset: Signal,
        experiment_config: ExperimentConfig,
    ):
        super().__init__(
            dataset=dataset,
            experiment_config=experiment_config,
            gt_example_image=dataset.signal_channel_first,
        )
        self.batch_size = self.experiment_config.batch_size

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.net(x)
        loss = F.mse_loss(y_pred, y)
        self.wandb_logger.log_metrics({"train_loss": loss}, step=self.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.experiment_config.optimizer.learning_rate
        )
        if self.experiment_config.optimizer.scheduler_name is None:
            return optimizer
        assert self.experiment_config.optimizer.scheduler_name == "CosineAnnealingLR"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **self.experiment_config.optimizer.scheduler_args
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self):
        self._validate_model(conserve_memory=True)

    def on_test_epoch_end(self):
        self._validate_model(conserve_memory=False)

    def _validate_model(self, conserve_memory=True):
        metrics = dict()

        _, out, out_train, out_test = self._infer_full_img()
        residuals = out - self.dataset.signal_channel_first
        train_residuals = out_train - self.dataset.train_pixels
        test_residuals = out_test - self.dataset.test_pixels

        metrics.update(
            self._get_metrics(self.dataset.signal_channel_first, out, residuals)
        )
        metrics["ssim"] = structural_similarity(
            self.dataset.signal_channel_first.numpy(),
            out.numpy(),
            channel_axis=0,
            data_range=2,
        )
        metrics.update(
            self._get_metrics(
                self.dataset.train_pixels, out_train, train_residuals, suffix="train"
            )
        )
        metrics.update(
            self._get_metrics(
                self.dataset.test_pixels, out_test, test_residuals, suffix="test"
            )
        )

        self._log_summary_table(metrics)
        self.wandb_logger.log_image(
            key="output",
            images=[tensor_to_image(out, limit_size=conserve_memory)],
            step=self.global_step,
            caption=["model_output"],
        )
        if self.global_step == 0:
            scaled_output = out - out.min()
            scaled_output = scaled_output / scaled_output.max()
            self.wandb_logger.log_image(
                key="scaled_output",
                images=[
                    tensor_to_image(scaled_output, limit_size=False, rescale=False)
                ],
                step=self.global_step,
                caption=["scaled_model_output"],
            )

    def _log_gt(self):
        self.wandb_logger.log_image(
            key="ground truth",
            images=[
                tensor_to_image(
                    self.dataset.signal_channel_first,
                )
            ],
            step=self.global_step,
        )

    def _infer_full_img(self):
        out = self.net.forward_in_batches(
            self.dataset.coords, batch_size=self.batch_size
        )
        out_train = out[self.dataset.train_idx]
        out_test = out[self.dataset.test_idx]
        img = out.reshape(self.dataset.shape)
        img = torch.movedim(img, -1, 0)  # move channels to the beginning
        return out, img, out_train, out_test

    @staticmethod
    def _get_metrics(gt, out, residuals, suffix=""):
        if suffix:
            suffix = f"_{suffix}"
        return {
            f"psnr{suffix}": peak_signal_noise_ratio(
                gt.numpy(), out.numpy(), data_range=2
            ),
            f"MAE{suffix}": torch.abs(residuals).mean(),
            f"max_residual{suffix}": torch.abs(residuals).max(),
        }


def inference_in_batches(model, x, batch_size=2**19, device="cuda"):
    """Run full inference and move results to cpu."""
    with torch.no_grad():
        out = []
        for i in range(0, x.shape[0], batch_size):
            out.append(model(x[i : i + batch_size].to(device=device)).to(device="cpu"))
        return torch.cat(out, dim=0)


def init_net_from_config(model_config: MLPConfig, dataset):
    if isinstance(model_config, SirenConfig):
        model = Siren(
            in_features=dataset.dims,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            out_features=dataset.channels,
            first_omega_0=model_config.omega_0,
            beta=model_config.beta,
            weight_constant=model_config.weight_constant,
            outermost_linear=True,
            apply_omega_when_linear=model_config.apply_omega_when_linear,
            boost_bias=model_config.boost_bias,
        )
        model.compile()
        return model
    elif isinstance(model_config, ReLUConfig):
        model = FourierFeaturesMLP(
            in_features=2,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            out_features=dataset.channels,
            sigma=model_config.sigma,
        )
        model.compile()
        return model
    elif isinstance(model_config, WireConfig):
        return Wire(
            in_features=2,
            out_features=dataset.channels,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            first_omega_0=model_config.omega_0,
            scale=model_config.scale,
            hidden_omega_0=model_config.hidden_omega,
        )
    elif isinstance(model_config, FinerConfig):
        model = Finer(
            in_features=2,
            out_features=dataset.channels,
            hidden_features=model_config.hidden_features,
            hidden_layers=model_config.hidden_layers,
            first_omega_0=model_config.omega_0,
            first_bias_scale=model_config.first_bias_scale,
            hidden_omega_0=model_config.hidden_omega,
        )
        model.compile()
        return model
    else:
        raise ValueError(f"Unknown model type: {model_config}")
