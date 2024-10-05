# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Union

import ray.train.torch
import torch
from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper, MaximumIterationStopper
from torch.utils.data import DataLoader

from torchlab import models
from torchlab.data import LabDataSet
from torchlab.utils import load_config


class Trainer:
    def __init__(
        self,
        # the training configs
        cfg_path: Union[Path, str],
        # torch settings
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        devices: int = torch.cuda.device_count(),
        torch_backend: str = "nccl" if torch.cuda.is_available() else "gloo",
        # stop settings
        max_iter: int = 100,
        plateau_metric: str = "val_loss",
        plateau_mode: str = "min",
        # checkpointing settings
        num_to_keep: int = 1,
        checkpoint_frequency: int = 50,
        checkpoint_score_attribute: str = "val_loss",
        checkpoint_score_order: str = "min",
        # scaling settings
        use_gpu: bool = torch.cuda.is_available(),
        cluster_workers: int = torch.cuda.device_count() if torch.cuda.is_available() else 1,
        # data settings
        dataloader_workers: int = mp.cpu_count() // 4,
    ) -> None:
        """a custom interface to Ray's TorchTrainer"""

        # the training configs
        self.cfg = load_config(cfg_path)
        # model
        self.modelcfg = {k: v for k, v in self.cfg.model.items() if k != "name"}
        _dims = {"sources": "pulses", "pulses": "sources"}
        # torch settings
        self.device = device
        self.devices = devices
        self.torch_backend = torch_backend
        # stop settings
        self.max_iter = max_iter
        self.plateau_metric = plateau_metric
        self.plateau_mode = plateau_mode
        # checkpointing settings
        self.num_to_keep = num_to_keep
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_score_attribute = checkpoint_score_attribute
        self.checkpoint_score_order = checkpoint_score_order
        # scaling settings
        self.cluster_workers = cluster_workers
        self.use_gpu = use_gpu
        # data settings
        self.dataloader_workers = dataloader_workers
        # logs settings
        self.storage_path = self.cfg.logs.ray
        # setup and validate config instantiation
        self._setup()

    def _setup(self):
        # #### LOGGER, STOPPERS #### #
        # https://docs.ray.io/en/latest/tune/examples/tune-wandb.html
        logger = WandbLoggerCallback(project=self.cfg.logger.project)
        # https://docs.ray.io/en/latest/tune/api/stoppers.html
        stoppers = CombinedStopper(
            MaximumIterationStopper(max_iter=self.max_iter),
            ExperimentPlateauStopper(metric=self.plateau_metric, mode=self.plateau_mode),
        )
        # ##### CONFIGS #### #
        # https://docs.ray.io/en/latest/train/getting-started-pytorch.html
        self.torch_config = TorchConfig(backend=self.torch_backend)
        # https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html#ray.train.CheckpointConfig
        # TorchTrainer does not support checkpoint_frequency or checkpoint_at_end
        checkpoint_config = CheckpointConfig(
            num_to_keep=self.num_to_keep,
            checkpoint_score_attribute=self.checkpoint_score_attribute,
            checkpoint_score_order=self.checkpoint_score_order,
        )
        # https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
        self.scaling_config = ScalingConfig(
            num_workers=self.cluster_workers,
            use_gpu=self.use_gpu,
        )
        # ##### RUN CONFIG #### #
        # https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig
        self.run_config = RunConfig(
            callbacks=[logger],
            checkpoint_config=checkpoint_config,
            stop=stoppers,
            storage_path=self.storage_path,
        )

    def _fit_loop(self, config) -> None:
        """fit loop per cluster worker"""

        # model
        model = getattr(models, self.cfg.model.name)
        model = model(batch_size=self.cfg.trainer.batch_size, **self.modelcfg)
        model = ray.train.torch.prepare_model(model, parallel_strategy=None if self.devices <= 1 else "ddp")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.L1Loss()
        # dataset and dataloader
        dataset = LabDataSet()
        dataloader = DataLoader(dataset, batch_size=self.cfg.trainer.batch_size, num_workers=self.dataloader_workers)
        dataloader = ray.train.torch.prepare_data_loader(dataloader)
        # logger
        wandb = setup_wandb(config, project=self.cfg.logger.project)

        for epoch in range(self.max_iter):

            if ray.train.get_context().get_world_size() > 1:
                dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(dataloader):
                x, y = batch
                output = model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_metrics = dict(step_loss=loss.item(), global_step=(step + 1) * (epoch + 1))
                wandb.log(step_metrics)

            # https://docs.ray.io/en/latest/train/user-guides/checkpoints.html
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                should_checkpoint = epoch % self.checkpoint_frequency == 0
                if ray.train.get_context().get_world_rank() == 0 and should_checkpoint:
                    torch.save(
                        model.state_dict(),  # NOTE: Unwrap the model.
                        os.path.join(temp_checkpoint_dir, "model.pt"),
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            epoch_metrics = {"epoch_loss": loss.item(), "epoch": epoch}
            wandb.log(epoch_metrics)
            ray.train.report(metrics={"checkpoint": checkpoint})

    def fit(self, return_results: bool = False) -> None | ray.train.Result:
        # ##### TORCH TRAINER #### #
        # https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html#ray.train.torch.TorchTrainer
        trainer = TorchTrainer(
            self._fit_loop,
            run_config=self.run_config,
            torch_config=self.torch_config,
            scaling_config=self.scaling_config,
        )
        # ##### FIT #### #
        result = trainer.fit()

        return result if return_results else None
