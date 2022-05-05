# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import torch
import torch.cuda.amp as amp
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from tqdm import tqdm

from torchsr.base.base_model import BaseModel
from torchsr.esrgan.discriminator import Discriminator
from torchsr.esrgan.generator import Generator
from torchsr.esrgan.loss import VGGLoss


class ESRGANTrainer(BaseModel):
    """
    A helper class to train ESRGAN models.

    Train an Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)
    to upscale input images of various sizes.

    Parameters
    ----------
    device : str
        A ``string`` of the primary device to use for computation, such as
        `cuda` for NVIDIA GPUs.
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.
    train_loader : DataLoader
        A ``DataLoader`` of all images in the training dataset.
    test_loader : DataLoader
        A ``DataLoader`` of all images in the testing dataset.
    train_len : int
        An ``int`` representing the total size of the training dataset.
    test_len : int
        An ``int`` representing the total size of the testing dataset.
    distributed : bool
        A ``boolean`` which evalutes to `True` if the training is distributed.
    """

    def _initialize_models(self) -> None:
        """
        Initialize the generator and discriminator models.
        """
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        if self.distributed:
            self.generator = DDP(
                self.generator,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            self.discriminator = DDP(
                self.discriminator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                # Required to avoid an error about inplace operations being
                # modified while running the forward pass of the discriminator
                # multiple times in one pass as is the case with the main
                # training loop.
                broadcast_buffers=False
            )

    def _initialize_loss(self) -> None:
        """
        Initialize all of the modules used to calculate loss.
        """
        self.l1_loss = nn.L1Loss().to(self.device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.vgg_loss = VGGLoss().to(self.device)
        self.val_loss = self.l1_loss

    def _pretrain(self) -> None:
        """
        Run the perceptual pre-training loop.

        Run the perceptual-based pre-training loop for the given number of
        epochs. The best recorded model from the pre-training phase will be
        used to initialize the weights of the generator in the second phase of
        training.
        """
        self._log('=' * 80)
        self._log('Starting pre-training')
        epoch = 1
        path = 'esrgan-psnr-latest.pth'

        if self.psnr_checkpoint:
            path = self.psnr_checkpoint

        checkpoint = self._load_checkpoint(path)
        if checkpoint:
            self.generator.load_state_dict(checkpoint["state"])
            epoch = checkpoint["epoch"]

        for epoch in range(epoch, self.pre_epochs + 1):
            step = 0

            self._log('-' * 80)
            self._log(f'Starting epoch {epoch} out of {self.pre_epochs}')

            self.generator.train()
            self.discriminator.train()

            start_time = time.time()

            for sub_step, (low_res, high_res) in enumerate(tqdm(self.train_loader, disable=not self.main_process)):
                high_res = high_res.to(self.device)
                low_res = low_res.to(self.device)

                self.psnr_optimizer.zero_grad()

                with amp.autocast(enabled=self.amp):
                    super_res = self.generator(low_res)
                    loss = self.l1_loss(super_res, high_res)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.psnr_optimizer)
                self.scaler.update()

                step = (sub_step * self.batch_size * self.world_size) + \
                       ((epoch - 1) * self.train_len)

                self._log_wandb(
                    {
                        'psnr/train-loss': loss,
                        'psnr/epoch': epoch
                    },
                    step=step
                )

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = len(self.train_loader) * self.batch_size * self.world_size / time_taken

            self._log(f'Throughput: {round(throughput, 3)} images/sec')
            self._log_wandb(
                {
                    'psnr/throughput/train': throughput,
                    'psnr/epoch': epoch
                },
                step=step
            )

            self._test(epoch, 'esrgan-psnr', step)

    def _gan_loop(self, low_res: Tensor, high_res: Tensor, step: int) -> None:
        """
        Run the main GAN-based training loop.

        Given low and high resolution input images, run the forward and
        backward passes of the model to train both the discriminator and the
        generator.

        Parameters
        ----------
        low_res : Tensor
            A ``tensor`` of a batch of low resolution images from the training
            dataset.
        high_res : Tensor
            A ``tensor`` of a batch of high resolution images from the training
            dataset.
        step : int
            An ``int`` of the current step the training loop is on.
        """
        low_res = low_res.to(self.device)
        high_res = high_res.to(self.device)
        batch_size = low_res.size(0)

        real_label = torch.full((batch_size, 1), 1, dtype=low_res.dtype).to(self.device)
        fake_label = torch.full((batch_size, 1), 0, dtype=low_res.dtype).to(self.device)

        self.disc_optimizer.zero_grad()

        with amp.autocast(enabled=self.amp):
            super_res = self.generator(low_res)
            real_output = self.discriminator(high_res)
            fake_output = self.discriminator(super_res.detach())

            disc_loss_real = self.bce_loss(real_output - torch.mean(fake_output), real_label)
            disc_loss_fake = self.bce_loss(fake_output - torch.mean(real_output), fake_label)
            disc_loss = (disc_loss_real + disc_loss_fake) / 2

        self.scaler.scale(disc_loss).backward()
        self.scaler.step(self.disc_optimizer)
        self.scaler.update()

        self.gen_optimizer.zero_grad()

        with amp.autocast(enabled=self.amp):
            super_res = self.generator(low_res)
            real_output = self.discriminator(high_res.detach())
            fake_output = self.discriminator(super_res)

            pixel_loss = self.l1_loss(super_res, high_res.detach())
            content_loss = self.vgg_loss(super_res, high_res.detach())
            adversarial_loss = self.bce_loss(fake_output - torch.mean(real_output), real_label)
            gen_loss = 0.01 * pixel_loss + 1 * content_loss + 0.005 * adversarial_loss

        self._log_wandb(
            {
                'gan/disc-lr': self.disc_scheduler.get_last_lr()[0],
                'gan/gen-lr': self.gen_scheduler.get_last_lr()[0],
                'gan/train-loss': gen_loss
            },
            step=step
        )

        self.scaler.scale(gen_loss).backward()
        self.scaler.step(self.gen_optimizer)
        self.scaler.update()

        self.generator.zero_grad()

    def _gan_train(self) -> None:
        """
        Run the training loop for the GAN phase.

        Iterate over each image in the training dataset and update the model
        over the requested number of epochs.
        """
        self._log('=' * 80)
        self._log('Starting training loop')
        epoch = 1

        self.best_psnr = -1.0
        try:
            if self.gan_checkpoint:
                path = self.gan_checkpoint
            else:
                path = 'esrgan-gan-latest.pth'
            checkpoint = self._load_checkpoint(path)
            # Prefer loading an existing GAN-based model before PSNR-based
            # model as a better base.
            if checkpoint:
                self.generator.load_state_dict(checkpoint["state"])
                epoch = checkpoint["epoch"]
            else:
                checkpoint = self._load_checkpoint('esrgan-psnr-latest.pth')
                if checkpoint:
                    self.generator.load_state_dict(checkpoint["state"])
        except FileNotFoundError:
            print('Pre-trained file not found. Training GAN from scratch.')

        for epoch in range(epoch, self.epochs + 1):
            step = 0

            self._log('-' * 80)
            self._log(f'Starting epoch {epoch} out of {self.epochs}')

            self.generator.train()
            self.discriminator.train()

            start_time = time.time()

            for sub_step, (low_res, high_res) in enumerate(tqdm(self.train_loader, disable=not self.main_process)):
                step = (sub_step * self.batch_size * self.world_size) + \
                       ((self.pre_epochs + epoch - 1) * self.train_len)
                self._gan_loop(low_res, high_res, step)

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = len(self.train_loader) * self.batch_size * self.world_size / time_taken

            self._log(f'Throughput: {round(throughput, 3)} images/sec')
            self._log_wandb(
                {
                    'gan/throughput/train': throughput,
                    'gan/epoch': epoch
                },
                step=step
            )
            
            self.disc_scheduler.step()
            self.gen_scheduler.step()

            self._test(epoch, 'esrgan-gan', step)
