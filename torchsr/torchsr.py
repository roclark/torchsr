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
                               MASTER_ADDR,
                               MASTER_PORT,
                               PRE_EPOCHS,
                               TRAIN_DIR)
from torchsr.dataset import initialize_datasets
from torchsr.test import test
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
        raise ArgumentTypeError('value must be a positive integer!')
    return int_value


def get_device(gpus: int) -> torch.device:
    """
    Return the device type to use.

    If the system has GPUs available and the user didn't explicitly specify 0
    GPUs, the device type will be 'cuda'. Otherwise, the device type will be
    'cpu' which will be significantly slower.

    Parameters
    ----------
    gpus : int
        An ``int`` of the number of GPUs to use during training.

    Returns
    -------
    torch.device
        Returns the specific type of device to use for computations where
        applicable.
    """
    if gpus > 0:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def gpu_count(gpus: int) -> int:
    """
    Find the number of GPUs to use.

    By default, the application attempts to use all GPUs available in the
    system but users can specify a different number of GPUs if desired. If the
    system doesn't have any GPUs available, it will default to the CPUs. If the
    user specifies a specific number of GPUs, it needs to be verified that it
    is a positive integer and less than or equal to the total number of GPUs
    available in the system.

    Parameters
    ----------
    gpus : int
        An ``int`` of the number of GPUs the user requested via the CLI,
        defaulting to 0.

    Returns
    -------
    int
        Returns an ``int`` of the number of GPUs to use in the system.
    """
    count = torch.cuda.device_count()
    if not torch.cuda.is_available():
        print('No GPUs available. Running on CPUs instead.')
        return 0
    if gpus is None:
        return count
    if gpus <= 0:
        print('No GPUs requested. Running on CPUs instead.')
        return 0
    if gpus <= count:
        return gpus
    if gpus > count:
        print(f'Requested {gpus} GPUs but only found {count}')
        print(f'Using {count} GPUs instead')
        return count
    return 0


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
    train.add_argument('--gpus', help='The number of GPUs to use for training '
                       'on a single system. The GPUs will be automatically '
                       'selected in numerical order. Default: All available '
                       'GPUs.', type=int, default=None)
    train.add_argument('--master-addr', help='The address to be used for all '
                       f'distributed communication. Default: {MASTER_ADDR}',
                       type=str, default=MASTER_ADDR)
    train.add_argument('--master-port', help='The port to use for all '
                       f'distributed communication. Default {MASTER_PORT}',
                       type=str, default=MASTER_PORT)
    train.add_argument('--pretrain-epochs', help='The number of epochs to '
                       f'run pretraining for. Default: {PRE_EPOCHS}.',
                       type=int, default=PRE_EPOCHS)
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
    return parser.parse_args()


def worker_environment(local_rank: int, world_size: int, args: Namespace) -> None:
    """
    Set the worker environment.

    In order to configure distributed training, the training environment needs
    to be set with various parameters including the master address and port to
    communicate on for multi-node, the local rank of the process, and the world
    size.

    Parameters
    ----------
    local_rank : int
        An ``int`` indicating which GPU to run on for a single node.
    world_size : int
        An ``int`` of the total number of processes for the entire cluster.
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.
    """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['OMP_NUM_THREADS'] = str(1)


def worker_process(local_rank: int, world_size: int, device: torch.device,
                   args: Namespace) -> None:
    """
    Initiate training in a new process.

    In each new process launched by the main function, setup the environment to
    communicate for distributed work and initialize the distributed data
    loaders before starting the training process.

    Parameters
    ----------
    local_rank : int
        An ``int`` denoting which GPU this process should run on.
    world_size : int
        An ``int`` of the total number of GPUs/processes the application will
        run on.
    device : torch.device
        The specific type of device to use for computations where applicable.
    args : Namespace
        A ``Namespace`` of all the arguments passed via the CLI.
    """
    worker_environment(local_rank, world_size, args)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.local_rank = local_rank
    train_loader, test_loader, train_len, test_len = initialize_datasets(
        args.train_dir,
        batch_size=args.batch_size,
        dataset_multiplier=args.dataset_multiplier,
        workers=args.data_workers,
        distributed=True
    )
    trainer = SRGANTrainer(device, args, train_loader, test_loader, train_len,
                           test_len, distributed=True)
    trainer.train()


def main() -> None:
    args = parse_args()
    gpus = gpu_count(args.gpus)
    device = get_device(gpus)
    distributed = gpus > 1
    
    if args.function == 'test':
        test(args, device)
    else:
        if distributed:
            # Hard-coded for single node at the moment.
            nodes = 1
            world_size = gpus * nodes
            torch.multiprocessing.spawn(
                worker_process,
                nprocs=gpus,
                args=(world_size, device, args)
            )
        else:
            train_loader, test_loader, train_len, test_len = initialize_datasets(
                args.train_dir,
                batch_size=args.batch_size,
                dataset_multiplier=args.dataset_multiplier,
                workers=args.data_workers,
                distributed=distributed
            )
            args.local_rank = -1
            trainer = SRGANTrainer(device, args, train_loader, test_loader,
                                   train_len, test_len)
            trainer.train()


if __name__ == '__main__':
    main()
