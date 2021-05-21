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
import torch.optim as optim
import torchvision.utils as utils
from argparse import Namespace
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Resize, ToTensor
from math import log10
from PIL import Image
from torch import nn
from tqdm import tqdm

from torchsr.discriminator import Discriminator
from torchsr.generator import Generator
from torchsr.loss import VGGLoss


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
        self.best_psnr = -1.0
        self.device = device
        self.distributed = distributed
        self.epochs = args.epochs
        self.local_rank = args.local_rank
        self.pre_epochs = args.pretrain_epochs
        self.save_image = not args.skip_image_save
        self.test_loader = test_loader
        self.test_len = test_len
        self.train_loader = train_loader
        self.train_len = train_len
        # For using a single process, the default rank is -1 for the first and
        # only process.
        self.main_process = args.local_rank in [-1, 0]

        if device == torch.device('cuda'):
            torch.cuda.set_device(args.local_rank)

        if SummaryWriter:
            self.writer = SummaryWriter()
        else:
            self.writer = None

        if self.save_image and self.main_process \
           and not os.path.exists('output'):
            os.makedirs('output')

        self._initialize_trainer()
        self._create_test_image()

    def _cleanup(self) -> None:
        """
        Remove and close any unnecessary items.
        """
        if self.writer:
            self.writer.close()

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
            step_size=self.epochs // 2,
            gamma=0.1
        )
        self.gen_scheduler = torch.optim.lr_scheduler.StepLR(
            self.gen_optimizer,
            step_size=self.epochs // 2,
            gamma=0.1
        )

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

    def _test(self, epoch: int, output: str) -> None:
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
        output : str
            A ``string`` of the current training phase.
        """
        self.generator.eval()

        self._log(f'Testing results after epoch {epoch}')

        with torch.no_grad():
            psnr = 0.0
            start_time = time.time()

            for low_res, _, high_res in tqdm(self.test_loader, disable=not self.main_process):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                super_res = self.generator(low_res).to(self.device)

                psnr += 10 * log10(1 / ((super_res - high_res) ** 2).mean().item())

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = self.test_len / time_taken
            psnr = psnr / len(self.test_loader)

            self._log(f'PSNR: {round(psnr, 3)}, '
                      f'Throughput: {round(throughput, 3)} images/sec')
            phase = output.rstrip('.pth')
            if self.writer:
                self.writer.add_scalar(f'{phase}/PSNR', psnr, epoch)
                self.writer.add_scalar(f'{phase}/throughput/test', throughput, epoch)

            if psnr > self.best_psnr:
                self.best_psnr = psnr
                torch.save(self.generator.state_dict(), output)

            # If the user requested to not save images, return immediately and
            # avoid generating and saving the image.
            if not self.save_image:
                return
            # Save a copy of a single image that has been super-resed for easy
            # tracking of progress.
            super_res = self.generator(self.test_image).to(self.device)
            utils.save_image(super_res, f'output/SR_epoch{epoch}.png', padding=5)
            _, _, height, width = super_res.shape
            output_image = Resize((height // 4, width // 4),
                                  interpolation=InterpolationMode.BICUBIC)(super_res)
            output_image = utils.make_grid(output_image)
            if self.writer:
                self.writer.add_image(f'images/epoch{epoch}', output_image)

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

        for epoch in range(1, self.pre_epochs + 1):
            self._log(f'Starting epoch {epoch} out of {self.pre_epochs}')

            self.generator.train()
            self.discriminator.train()

            start_time = time.time()

            for _, (low_res, high_res) in enumerate(tqdm(self.train_loader, disable=not self.main_process)):
                high_res = high_res.to(self.device)
                low_res = low_res.to(self.device)

                self.psnr_optimizer.zero_grad()

                super_res = self.generator(low_res)
                loss = self.mse_loss(super_res, high_res)
                loss.backward()
                self.psnr_optimizer.step()

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = self.train_len / time_taken

            self._log(f'Throughput: {round(throughput, 3)} images/sec')

            if self.writer:
                self.writer.add_scalar(f'psnr/throughput/train', throughput, epoch)
            self._test(epoch, 'psnr.pth')

    def _gan_loop(self, low_res: Tensor, high_res: Tensor) -> None:
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
        gen_loss = content_loss + 0.001 * adversarial_loss

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

        self.best_psnr = -1.0
        self.generator.load_state_dict(torch.load('psnr.pth'))

        for epoch in range(1, self.epochs + 1):
            self._log(f'Starting epoch {epoch} out of {self.epochs}')

            self.generator.train()
            self.discriminator.train()

            start_time = time.time()

            for _, (low_res, high_res) in enumerate(tqdm(self.train_loader, disable=not self.main_process)):
                self._gan_loop(low_res, high_res)

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = self.train_len / time_taken

            self._log(f'Throughput: {round(throughput, 3)} images/sec')

            if self.writer:
                self.writer.add_scalar(f'gan/throughput/train', throughput, epoch)
            
            self.disc_scheduler.step()
            self.gen_scheduler.step()

            self._test(epoch, 'gan.pth')

    def train(self) -> None:
        """
        Initiate the pre-training followed by the main training phases of the
        network.
        """
        self._pretrain()
        self._gan_train()
        self._cleanup()
