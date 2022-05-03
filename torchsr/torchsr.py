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
import torch
import torch.distributed as dist
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from torchsr.constants import (BATCH_SIZE,
                               EPOCHS,
                               MODEL,
                               PRE_EPOCHS,
                               TRAIN_DIR)
from torchsr.dataset import initialize_datasets
from torchsr.test import test
from torchsr.models import MODELS, select_test_model, select_trainer_model
from torchsr.__version__ import VERSION
from typing import Tuple


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
        raise ArgumentTypeError('value must be a positive integer!')
    return int_value


def get_device(args: Namespace) -> torch.device:
    """
    Return the device type to use.

    If the system has GPUs available and the user didn't explicitly specify 0
    GPUs, the device type will be 'cuda'. Otherwise, the device type will be
    'cpu' which will be significantly slower.

    Parameters
    ----------
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.

    Returns
    -------
    torch.device
        Returns the specific type of device to use for computations where
        applicable.
    """
    count = torch.cuda.device_count()
    if args.local_world_size > count:
        print('More processes per node requested than GPUs found')
        print('Assuming CPU-only mode...')
        return torch.device('cpu')
    elif count < 1 or not torch.cuda.is_available():
        print('No GPUs found')
        print('Running in CPU-only mode...')
        return torch.device('cpu')
    else:
        return torch.device('cuda')


def distributed_params(args: Namespace) -> Tuple[Namespace, bool]:
    """
    Parse the parameters for distributed training.

    The `torchrun` wrapper sets several environment variables that are required
    for distributed training. If any of these variables cannot be parsed
    properly, assume that the user did not request to run in distributed mode
    and only run on a single process.

    Parameters
    ----------
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.

    Returns
    -------
    Tuple
        Returns a ``tuple`` of the update arguments including the parsed or
        default values for distributed parameters as well as a boolean which
        evaluates to `True` when running in distributed mode.
    """
    try:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        if not args.master_addr:
            args.master_addr = str(os.environ['MASTER_ADDR'])
        if not args.master_port:
            args.master_port = str(os.environ['MASTER_PORT'])
        distributed = True
    except (KeyError, ValueError):
        # Check if the user called the application using Slurm and get the
        # values from Slurm.
        try:
            args.world_size = int(os.environ['SLURM_NTASKS'])
            args.rank = int(os.environ['SLURM_PROCID'])
            args.local_rank = int(os.environ['SLURM_LOCALID'])
            args.local_world_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            distributed = True
        except (KeyError, ValueError):
            # Distributed-mode was not called, so set the default values for
            # single worker mode.
            distributed = False
    if not distributed:
        args.world_size = 1
        args.rank = -1
        args.local_rank = -1
        args.local_world_size = 1
    return args, distributed


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
    commands = parser.add_subparsers(dest='function', metavar='function',
                                     required=True)
    train = commands.add_parser('train', help='Train an SRGAN model against an HD '
                                'dataset.')
    train.add_argument('--batch-size', help='The number of images to include '
                        f'in every batch. Default: {BATCH_SIZE}.', type=int,
                        default=BATCH_SIZE)
    train.add_argument('--data-workers', help='Specify the number of parallel '
                       'workers to spawn to read and preprocess data. In '
                       'general, the higher the number of workers, the faster '
                       'training and testing will be up to a certain point. A '
                       'good rule of thumb is to take the number of images in '
                       'the dataset, divide it by the batch size, and dividing '
                       'that by the number of GPUs being used, or 1 if '
                       'CPU-only while rounding up in both cases.', type=int,
                       default=16)
    train.add_argument('--dataset-multiplier', help='Artificially increase '
                        'the size of the dataset by taking N number of random '
                        'samples from each image in the training dataset. The '
                        'default behavior is to take a single random square '
                        'subsection of each image, but depending on the '
                        'size of the subsection and the overall image size, '
                        'this could ignore over 99%% of the image. To increase '
                        'the number of samples per image, use a multiplier '
                        'greater than 1.', type=positive_integer, default=1)
    train.add_argument('--disable-amp', help='Disable Automatic Mixed '
                       'Precision (AMP) which uses both float32 and float16 to'
                       ' boost performance. Disabling AMP can decrease '
                       'performance by as much as 2X or more.',
                       action='store_true')
    train.add_argument('--epochs', help='The number of epochs to run '
                        f'training for. Default: {EPOCHS}.', type=int,
                        default=EPOCHS)
    train.add_argument('--gan-checkpoint', help='Specify an existing trained '
                       'model for the GAN-based training phase.', type=str)
    train.add_argument('--master-addr', help='The address to be used for all '
                       f'distributed communication.', type=str)
    train.add_argument('--master-port', help='The port to use for all '
                       f'distributed communication.', type=str)
    train.add_argument('--model', help='Select the model to use for super '
                       'resolution.', type=str, default=MODEL,
                       choices=MODELS.keys())
    train.add_argument('--pretrain-epochs', help='The number of epochs to '
                       f'run pretraining for. Default: {PRE_EPOCHS}.',
                       type=int, default=PRE_EPOCHS)
    train.add_argument('--psnr-checkpoint', help='Specify an existing trained '
                       'model for the PSNR-based training phase.', type=str)
    train.add_argument('--skip-image-save', help='By default, a sample image '
                       'is generated after every epoch and saved to the '
                       '"outputs/" directory. Add this flag to skip generating '
                       'and saving the image to reduce disk space.',
                       action='store_true')
    train.add_argument('--train-dir', help='Specify the location to the '
                        'directory where training images are stored. Default: '
                        f'{TRAIN_DIR}.', type=str, default=TRAIN_DIR)

    test = commands.add_parser('test', help='Generated a super resolution '
                               'image based on a trained SRGAN model.')
    test.add_argument('image', type=str, help='Filename of image to upres.')
    test.add_argument('--model', help='Select the model to use for super '
                      'resolution.', choices=MODELS.keys(), type=str,
                      default=MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args, distributed = distributed_params(args)
    device = get_device(args)
    train_class, crop_size = select_trainer_model(args)
    
    if args.function == 'test':
        model = select_test_model(args)
        test(args, model, device)
    else:
        if distributed:
            dist.init_process_group(backend='nccl')
        train_loader, test_loader, train_len, test_len = initialize_datasets(
            args.train_dir,
            batch_size=args.batch_size,
            crop_size=crop_size,
            dataset_multiplier=args.dataset_multiplier,
            workers=args.data_workers,
            distributed=distributed
        )
        train_len *= args.world_size
        test_len *= args.world_size
        trainer = train_class(device, args, train_loader, test_loader,
                              train_len, test_len, distributed)
        trainer.train()


if __name__ == '__main__':
    main()
