import os
import re
import torch
import torch.optim as optim
import torchvision.utils as utils
from argparse import ArgumentParser
from math import log10
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import BATCH_SIZE, EPOCHS, TRAIN_DIR, VAL_DIR
from dataset import to_image, TrainData, ValData
from discriminator import Discriminator
from generator import Generator
from loss import VGGLoss
from version import VERSION


def parse_args():
    parser = ArgumentParser(f'torchSR Version: {VERSION}')
    parser.add_argument('--batch-size', help='The number of images to include '
                        f'in every batch. Default: {BATCH_SIZE}.', type=int,
                        default=BATCH_SIZE)
    parser.add_argument('--checkpoint', help='Specify the directory to load a '
                        'checkpoint from.', type=str)
    parser.add_argument('--epochs', help='The number of epochs to run '
                        f'training for. Default: {EPOCHS}.', type=int,
                        default=EPOCHS)
    parser.add_argument('--train-dir', help='Specify the location to the '
                        'directory where training images are stored. Default: '
                        f'{TRAIN_DIR}.', type=str, default=TRAIN_DIR)
    parser.add_argument('--val-dir', help='Specify the location to the '
                        'diredctory where validation images are stored. '
                        f'Default: {VAL_DIR}.', type=str, default=VAL_DIR)
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


def checkpoint_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def save_model(generator, discriminator, optimizer_gen, optimizer_dis, epoch, device):
    directory = checkpoint_dir(f'checkpoints/epoch{epoch}')
    torch.save(generator.state_dict(), f'{directory}/generator_{device}.pth')
    torch.save(discriminator.state_dict(), f'{directory}/discriminator_{device}.pth')
    torch.save(optimizer_gen.state_dict(), f'{directory}/optimizer_gen_{device}.pth')
    torch.save(optimizer_dis.state_dict(), f'{directory}/optimizer_dis_{device}.pth')


def load_model(checkpoint, generator, discriminator, optimizer_gen, optimizer_dis, device):
    if checkpoint:
        generator.load_state_dict(torch.load(f'{checkpoint}/generator_{device}.pth'))
        discriminator.load_state_dict(torch.load(f'{checkpoint}/discriminator_{device}.pth'))
        optimizer_gen.load_state_dict(torch.load(f'{checkpoint}/optimizer_gen_{device}.pth'))
        optimizer_dis.load_state_dict(torch.load(f'{checkpoint}/optimizer_dis_{device}.pth'))


def train_psnr(dataloader, generator, mse_loss, optimizer, epoch, device, args):
    print(f'Starting epoch {epoch} out of {args.epochs}')

    for index, (low_res, high_res) in enumerate(tqdm(dataloader)):
        high_res = high_res.to(device)
        low_res = low_res.to(device)

        optimizer.zero_grad()

        super_res = generator(low_res)
        loss = mse_loss(super_res, high_res)
        loss.backward()
        optimizer.step()


def train_gan(dataloader, generator, gen_optimizer, discriminator, disc_optimizer, vgg_loss, bce_loss, epoch, device, args):
    print(f'Starting epoch {epoch} out of {args.epochs}')

    for index, (low_res, high_res) in enumerate(tqdm(dataloader)):
        high_res.to(device)
        low_res.to(device)
        batch_size = low_res.size(0)

        real_label = torch.full((batch_size, 1), 1, dtype=low_res.dtype).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=low_res.dtype).to(device)

        discriminator.zero_grad()

        super_res = generator(low_res)

        disc_loss_real = bce_loss(discriminator(high_res), real_label)
        disc_loss_fake = bce_loss(discriminator(super_res.detach()), fake_label)
        disc_loss = disc_loss + disc_loss_fake

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

    train_data = TrainData(args.train_dir, crop_size=128, upscale_factor=4)
    train_loader = DataLoader(dataset=train_data, num_workers=2, batch_size=args.batch_size, shuffle=True)
    val_data = ValData(args.val_dir, upscale_factor=4)
    val_loader = DataLoader(dataset=val_data, num_workers=1, batch_size=1, shuffle=False)

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        train_psnr(train_loader, generator, mse_loss, psnr_optimizer, epoch, device, args)
        validate(generator, val_loader, epoch, device, output='psnr.pth')

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        train_gan(train_loader, generator, gen_optimizer, discriminator, disc_optimizer, vgg_loss, bce_loss, epoch, device, args)
        validate(generator, val_loader, epoch, device, output='gan.pth')


if __name__ == "__main__":
    main()
