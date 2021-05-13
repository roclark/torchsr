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
import torch
import torch.optim as optim
import torchvision.utils as utils
from argparse import Namespace
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToPILImage, ToTensor
from math import log10
from PIL import Image
from torch import nn
from tqdm import tqdm

from discriminator import Discriminator
from generator import Generator
from loss import VGGLoss


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
    """
    def __init__(self, device: str, args: Namespace, train_loader: DataLoader,
                 test_loader: DataLoader) -> None:
        self.best_psnr = -1.0
        self.device = device
        self.epochs = args.epochs
        self.pre_epochs = args.pretrain_epochs
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.writer = SummaryWriter()

        self._initialize_trainer()
        self._create_test_image()

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

        print(f'Testing results after epoch {epoch}')

        with torch.no_grad():
            psnr = 0.0

            for low_res, _, high_res in tqdm(self.test_loader):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                super_res = self.generator(low_res).to(self.device)

                psnr += 10 * log10(1 / ((super_res - high_res) ** 2).mean().item())
            psnr = psnr / len(self.test_loader)
            print(f'PSNR: {round(psnr, 3)}')
            phase = output.rstrip('.pth')
            self.writer.add_scalar(f'{phase}/PSNR', psnr, epoch)

            if psnr > self.best_psnr:
                self.best_psnr = psnr
                torch.save(self.generator.state_dict(), output)

            # Save a copy of a single image that has been super-resed for easy
            # tracking of progress.
            super_res = self.generator(self.test_image).to(self.device)
            utils.save_image(super_res, f'output/SR_epoch{epoch}.png', padding=5)
            output_image = utils.make_grid(super_res)
            self.writer.add_image(f'images/epoch{epoch}', output_image)

    def _pretrain(self) -> None:
        """
        Run the perceptual pre-training loop.

        Run the perceptual-based pre-training loop for the given number of
        epochs. The best recorded model from the pre-training phase will be
        used to initialize the weights of the generator in the second phase of
        training.
        """
        print('=' * 80)
        print('Starting pre-training')

        for epoch in range(1, self.pre_epochs + 1):
            print(f'Starting epoch {epoch} out of {self.pre_epochs}')

            self.generator.train()
            self.discriminator.train()

            for _, (low_res, high_res) in enumerate(tqdm(self.train_loader)):
                high_res = high_res.to(self.device)
                low_res = low_res.to(self.device)

                self.psnr_optimizer.zero_grad()

                super_res = self.generator(low_res)
                loss = self.mse_loss(super_res, high_res)
                loss.backward()
                self.psnr_optimizer.step()

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
        print('=' * 80)
        print('Starting training loop')

        self.best_psnr = -1.0
        self.generator.load_state_dict(torch.load('psnr.pth'))

        for epoch in range(1, self.epochs + 1):
            print(f'Starting epoch {epoch} out of {self.epochs}')

            self.generator.train()
            self.discriminator.train()

            for _, (low_res, high_res) in enumerate(tqdm(self.train_loader)):
                self._gan_loop(low_res, high_res)
            
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
