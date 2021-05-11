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
from argparse import ArgumentParser
from math import log10
from torch import nn
from tqdm import tqdm

from constants import BATCH_SIZE, EPOCHS, PRE_EPOCHS, TRAIN_DIR, TEST_DIR
from dataset import to_image, test_dataset, train_dataset
from discriminator import Discriminator
from generator import Generator
from loss import VGGLoss
from version import VERSION


def parse_args():
    parser = ArgumentParser(f'torchSR Version: {VERSION}')
    parser.add_argument('--batch-size', help='The number of images to include '
                        f'in every batch. Default: {BATCH_SIZE}.', type=int,
                        default=BATCH_SIZE)
    parser.add_argument('--epochs', help='The number of epochs to run '
                        f'training for. Default: {EPOCHS}.', type=int,
                        default=EPOCHS)
    parser.add_argument('--pretrain-epochs', help='The number of epochs to '
                        'run pretraining for. Default: {PRE_EPOCHS}.',
                        type=int, default=PRE_EPOCHS)
    parser.add_argument('--train-dir', help='Specify the location to the '
                        'directory where training images are stored. Default: '
                        f'{TRAIN_DIR}.', type=str, default=TRAIN_DIR)
    parser.add_argument('--test-dir', help='Specify the location to the '
                        'diredctory where validation images are stored. '
                        f'Default: {TEST_DIR}.', type=str, default=TEST_DIR)
    return parser.parse_args()


def validate(generator, val_loader, epoch, device, output):
    val_images = []
    generator.eval()
    best_psnr = -1.0

    print(f'Validating results after epoch {epoch}')

    with torch.no_grad():
        psnr = 0

        for low_res, bicubic, high_res in tqdm(val_loader):
            batch_size = low_res.size(0)
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            super_res = generator(low_res).to(device)

            psnr += 10 * log10(1 / ((super_res - high_res) ** 2).mean().item())
            val_images.extend([to_image()(bicubic.data.cpu().squeeze(0)),
                               to_image()(high_res.data.cpu().squeeze(0)),
                               to_image()(super_res.data.cpu().squeeze(0))])

        psnr = psnr / len(val_loader)
        print(f'PSNR: {round(psnr, 3)}')

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(generator.state_dict(), output)

        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 3)

        for index, image in enumerate(val_images):
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, f'output/epoch_{epoch}_val_{index}.png', padding=5)


def train_psnr(dataloader, generator, mse_loss, optimizer, epoch, device, args):
    print(f'Starting epoch {epoch} out of {args.pretrain_epochs}')

    for _, (low_res, high_res) in enumerate(tqdm(dataloader)):
        high_res = high_res.to(device)
        low_res = low_res.to(device)

        optimizer.zero_grad()

        super_res = generator(low_res)
        loss = mse_loss(super_res, high_res)
        loss.backward()
        optimizer.step()


def train_gan(dataloader, generator, gen_optimizer, discriminator, disc_optimizer, vgg_loss, bce_loss, epoch, device, args):
    print(f'Starting epoch {epoch} out of {args.epochs}')

    for _, (low_res, high_res) in enumerate(tqdm(dataloader)):
        high_res = high_res.to(device)
        low_res = low_res.to(device)
        batch_size = low_res.size(0)

        real_label = torch.full((batch_size, 1), 1, dtype=low_res.dtype).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=low_res.dtype).to(device)

        discriminator.zero_grad()

        super_res = generator(low_res)

        disc_loss_real = bce_loss(discriminator(high_res), real_label)
        disc_loss_fake = bce_loss(discriminator(super_res.detach()), fake_label)
        disc_loss = disc_loss_real + disc_loss_fake

        disc_loss.backward()
        disc_optimizer.step()

        generator.zero_grad()

        content_loss = vgg_loss(super_res, high_res.detach())
        adversarial_loss = bce_loss(discriminator(super_res), real_label)
        gen_loss = content_loss + 0.001 * adversarial_loss

        gen_loss.backward()
        gen_optimizer.step()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    mse_loss = nn.MSELoss().to(device)
    bce_loss = nn.BCELoss().to(device)
    vgg_loss = VGGLoss().to(device)

    psnr_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=args.epochs // 2, gamma=0.1)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=args.epochs // 2, gamma=0.1)

    train_loader = train_dataset(args.train_dir, batch_size=args.batch_size)
    test_loader = test_dataset(args.test_dir)

    for epoch in range(1, args.pretrain_epochs + 1):
        generator.train()
        discriminator.train()

        train_psnr(train_loader, generator, mse_loss, psnr_optimizer, epoch, device, args)
        validate(generator, test_loader, epoch, device, output='psnr.pth')

    generator.load_state_dict(torch.load('psnr.pth'))

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        train_gan(train_loader, generator, gen_optimizer, discriminator, disc_optimizer, vgg_loss, bce_loss, epoch, device, args)

        disc_scheduler.step()
        gen_scheduler.step()

        validate(generator, test_loader, epoch, device, output='gan.pth')


if __name__ == "__main__":
    main()
