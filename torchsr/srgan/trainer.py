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
import os
import time
import torch
import torch.cuda.amp as amp
import torch.optim as optim
import torchvision.utils as utils
from argparse import Namespace
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Resize, ToTensor
from math import log10
from PIL import Image
from torch import nn
from tqdm import tqdm

from torchsr.srgan.discriminator import Discriminator
from torchsr.srgan.generator import Generator
from torchsr.srgan.loss import VGGLoss


class SRGANTrainer:
    """
    A helper class to train SRGAN models.

    Train a Super-Resolution Generative Adversarial Network (SRGAN) to upscale
    input images of various sizes.

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
    def __init__(self, device: str, args: Namespace, train_loader: DataLoader,
                 test_loader: DataLoader, train_len: int, test_len: int,
                 distributed: bool = False) -> None:
        self.amp = not args.disable_amp
        self.batch_size = args.batch_size
        self.best_psnr = -1.0
        self.device = device
        self.distributed = distributed
        self.epochs = args.epochs
        self.gan_checkpoint = args.gan_checkpoint
        self.local_rank = args.local_rank
        self.pre_epochs = args.pretrain_epochs
        self.psnr_checkpoint = args.psnr_checkpoint
        self.save_image = not args.skip_image_save
        self.test_loader = test_loader
        self.test_len = test_len
        self.train_loader = train_loader
        self.train_len = train_len
        self.world_size = args.world_size
        # For using a single process, the default rank is -1 for the first and
        # only process.
        self.main_process = args.rank in [-1, 0]

        if device == torch.device('cuda'):
            torch.cuda.set_device(args.local_rank)

        if self.save_image and self.main_process \
           and not os.path.exists('output'):
            os.makedirs('output')

        self._initialize_trainer()
        self._create_test_image()

    def _cleanup(self) -> None:
        """
        Remove and close any unnecessary items.
        """
        if wandb:
            wandb.finish()

    def _load_checkpoint(self, path: str) -> dict:
        """
        Load a pre-trained checkpoint of a specific name.

        Check if a checkpoint exists, and if so, load and return the checkpoint.
        If no checkpoint is found, return None.

        Parameters
        ----------
        path : str
            A ``string`` of the path to load the checkpoint from.

        Returns
        -------
        dict
            Returns a ``dict`` of the loaded model and metadata. Returns `None`
            if no checkpoint can be found.
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            return checkpoint
        else:
            return None

    def _create_test_image(self) -> None:
        """
        Load the test image to be used to verify the model after every epoch.
        """
        image = Image.open('media/waterfalls-low-res.png')
        image = ToTensor()(image)
        self.test_image = image.unsqueeze(0).to(self.device)

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
        self.mse_loss = nn.MSELoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)
        self.vgg_loss = VGGLoss().to(self.device)

    def _initialize_optimizers(self) -> None:
        """
        Initialize the optimizers and schedulers.
        """
        self.psnr_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999)
        )
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999)
        )
        self.disc_scheduler = torch.optim.lr_scheduler.StepLR(
            self.disc_optimizer,
            step_size=self.epochs // 8,
            gamma=0.6
        )
        self.gen_scheduler = torch.optim.lr_scheduler.StepLR(
            self.gen_optimizer,
            step_size=self.epochs // 8,
            gamma=0.6
        )
        self.scaler = amp.GradScaler(enabled=self.amp)

    def _initialize_trainer(self) -> None:
        """
        Setup the SRGAN trainer by initializing all models, loss functions,
        optimizers, and schedulers.
        """
        self._initialize_models()
        self._initialize_loss()
        self._initialize_optimizers()

    def _log(self, statement: str) -> None:
        """
        Print a statement only on the main process.

        Parameters
        ----------
        statement : str
            A ``string`` to print out only on the main process.
        """
        if self.main_process:
            print(statement)

    def _log_wandb(self, contents: dict, step: int = None) -> None:
        """
        Log a dictionary of contents to Weights and Biases if configured.

        Parameters
        ----------
        contents : dict
            A ``dictionary`` of information to log to Weights and Biases.
        step : int
            An ``int`` of the current step.
        """
        if wandb and self.main_process:
            wandb.log(contents, step=step)

    def _model_state(self, epoch: int, phase: str) -> dict:
        """
        Create a model save state with metadata.

        Various metadata points related to training are valuable to be included
        in the checkpoint to make it easier to pickup progress where it was
        left off while continuing from an existing checkpoint.

        Parameters
        ----------
        epoch : int
            An ``int`` of the current epoch in the training pass.
        phase : str
            A ``string`` of the current training phase.

        Returns
        -------
        dict
            Returns a ``dict`` of the latest model state including metadata
            information.
        """
        return {
            "epoch": epoch,
            "phase": phase,
            "state": self.generator.state_dict()
        }

    def _test(self, epoch: int, phase: str, step: int) -> None:
        """
        Run a test pass against the test dataset and sample image.

        After every epoch, run through the test dataset to generate a super
        resolution version of every image based on the latest model weights and
        calculate the average PSNR for the dataset. If the PSNR is a new
        record, the model will be saved to allow re-training.

        After iterating through the test dataset, a super resolution version of
        a default image is generated and saved locally to compare results over
        time.

        Parameters
        ----------
        epoch : int
            An ``int`` of the current epoch in the training pass.
        phase : str
            A ``string`` of the current training phase.
        step : int
            An ``int`` of the current step in the training pass.
        """
        self.generator.eval()

        self._log(f'Testing results after epoch {epoch}')

        with torch.no_grad():
            loss, psnr = 0.0, 0.0
            start_time = time.time()

            for low_res, _, high_res in tqdm(self.test_loader, disable=not self.main_process):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                super_res = self.generator(low_res).to(self.device)

                psnr += 10 * log10(1 / ((super_res - high_res) ** 2).mean().item())

                loss += self.mse_loss(super_res, high_res)

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = len(self.test_loader) * self.batch_size * self.world_size / time_taken
            psnr = psnr / len(self.test_loader)
            loss = loss / len(self.test_loader)

            self._log(f'PSNR: {round(psnr, 3)}, '
                      f'Throughput: {round(throughput, 3)} images/sec')

            # Strip so we get just the short phase, ie 'gan' or 'psnr'
            short_phase = ''.join(phase.split('-')[1:])
            self._log_wandb(
                {
                    f'{short_phase}/PSNR': psnr,
                    f'{short_phase}/val-pixel-loss': loss,
                    f'{short_phase}/throughput/test': throughput,
                    f'{short_phase}/epoch': epoch
                },
                step=step
            )

            if psnr > self.best_psnr and self.main_process:
                self.best_psnr = psnr
                torch.save(self._model_state(epoch, phase),
                           f'{phase}-best.pth')
            if self.main_process:
                torch.save(self._model_state(epoch, phase),
                           f'{phase}-latest.pth')

            # If the user requested to not save images, return immediately and
            # avoid generating and saving the image.
            if not self.save_image:
                return
            # Save a copy of a single image that has been super-resed for easy
            # tracking of progress.
            super_res = self.generator(self.test_image).to(self.device)
            utils.save_image(super_res, f'output/SR_epoch{epoch}.png', padding=5)
            _, _, height, width = super_res.shape
            super_res_resize = Resize((height // 4, width // 4),
                                  interpolation=InterpolationMode.BICUBIC)(super_res)
            if wandb:
                self._log_wandb(
                    {f'images/epoch{epoch}': wandb.Image(super_res_resize)}
                )

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
        path = 'srgan-psnr-latest.pth'

        if self.psnr_checkpoint:
            path = self.psnr_checkpoint

        checkpoint = self._load_checkpoint(path)
        if checkpoint:
            self.generator.load_state_dict(checkpoint["state"])
            epoch = checkpoint["epoch"]

        for epoch in range(epoch, self.pre_epochs + 1):
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
                    loss = self.mse_loss(super_res, high_res)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.psnr_optimizer)
                self.scaler.update()

                step = (sub_step * self.batch_size * self.world_size) + \
                       ((epoch - 1) * self.train_len)

                self._log_wandb(
                    {
                        'psnr/train-pixel-loss': loss,
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

            self._test(epoch, 'srgan-psnr', step)

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

        self.discriminator.zero_grad()

        super_res = self.generator(low_res)

        disc_loss_real = self.bce_loss(self.discriminator(high_res), real_label)
        disc_loss_fake = self.bce_loss(self.discriminator(super_res.detach()), fake_label)
        disc_loss = disc_loss_real + disc_loss_fake

        disc_loss.backward()
        self.disc_optimizer.step()

        self.generator.zero_grad()

        content_loss = self.vgg_loss(super_res, high_res.detach())
        adversarial_loss = self.bce_loss(self.discriminator(super_res), real_label)
        gen_loss = content_loss + adversarial_loss

        self._log_wandb(
            {
                'gan/disc-lr': self.disc_scheduler.get_last_lr()[0],
                'gan/gen-lr': self.gen_scheduler.get_last_lr()[0],
                'gan/train-content-loss': content_loss,
                'gan/train-adversarial-loss': adversarial_loss,
                'gan/train-generator-loss': gen_loss
            },
            step=step
        )

        gen_loss.backward()
        self.gen_optimizer.step()

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
                path = 'srgan-gan-latest.pth'
            checkpoint = self._load_checkpoint(path)
            # Prefer loading an existing GAN-based model before PSNR-based
            # model as a better base.
            if checkpoint:
                self.generator.load_state_dict(checkpoint["state"])
                epoch = checkpoint["epoch"]
            else:
                checkpoint = self._load_checkpoint('srgan-psnr-latest.pth')
                if checkpoint:
                    self.generator.load_state_dict(checkpoint["state"])
        except FileNotFoundError:
            print('Pre-trained file not found. Training GAN from scratch.')

        for epoch in range(epoch, self.epochs + 1):
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

            self._test(epoch, 'srgan-gan', step)

    def train(self) -> None:
        """
        Initiate the pre-training followed by the main training phases of the
        network.
        """
        self._pretrain()
        # Clear the GPU cache between pre-training and main training phases to
        # optimize memory usage during extended runtime.
        torch.cuda.empty_cache()
        self._gan_train()
        self._cleanup()
