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
from argparse import ArgumentParser

from constants import BATCH_SIZE, EPOCHS, PRE_EPOCHS, TRAIN_DIR, TEST_DIR
from dataset import test_dataset, train_dataset
from trainer import SRGANTrainer
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


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = train_dataset(args.train_dir, batch_size=args.batch_size)
    test_loader = test_dataset(args.test_dir)

    trainer = SRGANTrainer(device, args, train_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()
