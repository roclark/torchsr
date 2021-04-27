import torch
import torch.optim as optim
import torchvision.utils as utils
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import BATCH_SIZE, EPOCHS, TRAIN_DIR, VAL_DIR
from dataset import to_image, TrainData, ValData
from models import Discriminator, Generator
from version import VERSION


def parse_args():
    parser = ArgumentParser(f'torchSR Version: {VERSION}')
    parser.add_argument('--batch-size', help='The number of images to include '
                        f'in every batch. Default: {BATCH_SIZE}.', type=int,
                        default=BATCH_SIZE)
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


def validate(generator, val_loader, epoch, device):
    val_images = []

    print(f'Validating results after epoch {epoch}')

    with torch.no_grad():
        generator.eval()

        for low_res_val, ground_truth_restore, ground_truth_val in tqdm(val_loader):
            batch_size = low_res_val.size(0)
            low_res = low_res_val.to(device)
            high_res = ground_truth_val.to(device)

            super_res = generator(low_res).to(device)
            val_images.extend([to_image()(ground_truth_restore.squeeze(0)),
                               to_image()(high_res.data.cpu().squeeze(0)),
                               to_image()(super_res.data.cpu().squeeze(0))])

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


def train(generator, discriminator, optimizer_gen, optimizer_dis, mse, bce,
          train_loader, epoch, device, args):
    print(f'Starting epoch {epoch} out of {args.epochs}')

    for low_res, ground_truth in tqdm(train_loader):
        ground_truth = ground_truth.to(device)
        low_res = low_res.to(device)

        discriminator.zero_grad()
        logits_real = discriminator(ground_truth)
        logits_fake = discriminator(generator(low_res).detach())
        real = torch.tensor(torch.rand(logits_real.size()) * 0.25 + 0.85).to(device)
        fake = torch.tensor(torch.rand(logits_fake.size()) * 0.15).to(device)
        prob = (torch.rand(logits_real.size()) < 0.05).to(device)

        real_clone = real.clone()
        real[prob] = fake[prob]
        fake[prob] = real_clone[prob]
        dis_loss = bce(logits_real, real) + bce(logits_fake, fake)
        dis_loss.backward()
        optimizer_dis.step()

        generator.zero_grad()
        fake_high_res = generator(low_res)
        image_loss = mse(fake_high_res, ground_truth)

        logits_fake_new = discriminator(fake_high_res)
        adversarial_loss = bce(logits_fake_new, torch.ones_like(logits_fake_new))

        gradient_loss = image_loss + 1e-2 * adversarial_loss
        gradient_loss.backward()
        optimizer_gen.step()
    save_model(generator, discriminator, optimizer_gen, optimizer_dis, epoch, device)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    mse = nn.MSELoss().to(device)
    bce = nn.BCELoss().to(device)

    train_data = TrainData(args.train_dir, crop_size=128, upscale_factor=4)
    train_loader = DataLoader(dataset=train_data, num_workers=2, batch_size=args.batch_size, shuffle=True)
    val_data = ValData(args.val_dir, upscale_factor=4)
    val_loader = DataLoader(dataset=val_data, num_workers=1, batch_size=1, shuffle=False)

    optimizer_gen = optim.Adam(generator.parameters())
    optimizer_dis = optim.Adam(discriminator.parameters())

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        train(generator, discriminator, optimizer_gen, optimizer_dis, mse, bce,
              train_loader, epoch, device, args)
        validate(generator, val_loader, epoch, device)


if __name__ == "__main__":
    main()
