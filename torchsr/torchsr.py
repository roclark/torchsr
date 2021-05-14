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
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from torchsr.constants import BATCH_SIZE, EPOCHS, PRE_EPOCHS, TRAIN_DIR
from torchsr.dataset import initialize_datasets
from torchsr.trainer import SRGANTrainer
from torchsr.__version__ import VERSION


def positive_integer(value: str) -> int:
    """
    Determine if a number is a positive integer.

    Some arguments need to be a positive integer to function properly. Check
    if the input is a valid number and force it to an integer before checking
    if it is greater than zero. If any checks fail, raise an ArgumentTypeError.

    Parameters
    ----------
    value : str
        A ``string`` of the input passed by the user.

    Returns
    -------
    int
        Returns an ``int`` of the positive integer passed by the user.

    Raises
    ------
    ArgumentTypeError
        Raises an ``ArgumentTypeError`` if the input is not an integer or if
        the number is not greater than zero.
    """
    try:
        int_value = int(value)
    except TypeError:
        raise ArgumentTypeError(f'invalid int value: \'{value}\'')
    if int_value < 1:
        raise ArgumentTypeError('dataset-multiplier must be a positive '
                                'integer!')
    return int_value


def parse_args() -> Namespace:
    """
    Parse arguments.

    Parse optional arguments passed to the application during runtime and
    return the results.

    Returns
    -------
    Namespace
        Returns a ``Namespace`` containing all of the arguments passed by the
        user including defaults.
    """
    parser = ArgumentParser(f'torchSR Version: {VERSION}')
    parser.add_argument('--batch-size', help='The number of images to include '
                        f'in every batch. Default: {BATCH_SIZE}.', type=int,
                        default=BATCH_SIZE)
    parser.add_argument('--dataset-multiplier', help='Artificially increase '
                        'the size of the dataset by taking N number of random '
                        'samples from each image in the training dataset. The '
                        'default behavior is to take a single random square '
                        'subsection of each image, but depending on the '
                        'size of the subsection and the overall image size, '
                        'this could ignore over 99%% of the image. To increase '
                        'the number of samples per image, use a multiplier '
                        'greater than 1.', type=positive_integer, default=1)
    parser.add_argument('--epochs', help='The number of epochs to run '
                        f'training for. Default: {EPOCHS}.', type=int,
                        default=EPOCHS)
    parser.add_argument('--pretrain-epochs', help='The number of epochs to '
                        'run pretraining for. Default: {PRE_EPOCHS}.',
                        type=int, default=PRE_EPOCHS)
    parser.add_argument('--train-dir', help='Specify the location to the '
                        'directory where training images are stored. Default: '
                        f'{TRAIN_DIR}.', type=str, default=TRAIN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = initialize_datasets(
        args.train_dir,
        batch_size=args.batch_size,
        dataset_multiplier=args.dataset_multiplier
    )

    trainer = SRGANTrainer(device, args, train_loader, test_loader)
    trainer.train()


if __name__ == '__main__':
    main()
