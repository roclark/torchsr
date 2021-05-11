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
from math import log10
from torch import nn
from tqdm import tqdm

from dataset import to_image
from discriminator import Discriminator
from generator import Generator
from loss import VGGLoss


class SRGANTrainer:
    def __init__(self, device, args, train_loader, test_loader):
        self.best_psnr = -1.0
        self.device = device
        self.epochs = args.epochs
        self.pre_epochs = args.pretrain_epochs
        self.test_loader = test_loader
        self.train_loader = train_loader

        self._initialize_trainer()

    def _initialize_models(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

    def _initialize_loss(self):
        self.mse_loss = nn.MSELoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)
        self.vgg_loss = VGGLoss().to(self.device)

    def _initialize_optimizers(self):
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

    def _initialize_trainer(self):
        self._initialize_models()
        self._initialize_loss()
        self._initialize_optimizers()

    def _test(self, epoch, output):
        test_images = []
        self.generator.eval()

        print(f'Testing results after epoch {epoch}')

        with torch.no_grad():
            psnr = 0.0

            for low_res, bicubic, high_res in tqdm(self.test_loader):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                super_res = self.generator(low_res).to(self.device)

                psnr += 10 * log10(1 / ((super_res - high_res) ** 2).mean().item())
                test_images.extend([to_image()(bicubic.data.cpu().squeeze(0)),
                                    to_image()(high_res.data.cpu().squeeze(0)),
                                    to_image()(super_res.data.cpu().squeeze(0))])

            psnr = psnr / len(self.test_loader)
            print(f'PSNR: {round(psnr, 3)}')

            if psnr > self.best_psnr:
                self.best_psnr = psnr
                torch.save(self.generator.state_dict(), output)

            test_images = torch.stack(test_images)
            test_images = torch.chunk(test_images, test_images.size(0) // 3)

            for index, image in enumerate(test_images):
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, f'output/epoch_{epoch}_val_{index}.png', padding=5)

    def _pretrain(self):
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

    def _gan_loop(self, low_res, high_res):
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

    def _gan_train(self):
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

    def train(self):
        self._pretrain()
        self._gan_train()
