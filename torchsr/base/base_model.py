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
import torch.cuda.amp as amp
import torch.optim as optim
import torchvision.utils as utils
from argparse import Namespace
from math import log10
from PIL import Image
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Resize, ToTensor
from tqdm import tqdm


class BaseModel:
    """
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
        self.amp = not args.disable_amp
        self.batch_size = args.batch_size
        self.best_psnr = -1.0
        self.device = device
        self.distributed = distributed
        self.epochs = args.epochs
        self.gan_checkpoint = args.gan_checkpoint
        self.local_rank = args.local_rank
        self.pre_epochs = args.pretrain_epochs
        self.psnr_checkpoint = args.psnr_checkpoint
        self.save_image = not args.skip_image_save
        self.test_loader = test_loader
        self.test_len = test_len
        self.train_loader = train_loader
        self.train_len = train_len
        self.world_size = args.world_size
        # For using a single process, the default rank is -1 for the first and
        # only process.
        self.main_process = args.rank in [-1, 0]

        if device == torch.device('cuda'):
            torch.cuda.set_device(args.local_rank)

        if self.save_image and self.main_process \
           and not os.path.exists('output'):
            os.makedirs('output')

        self._initialize_trainer()
        self._create_test_image()

    def _cleanup(self) -> None:
        """
        Remove and close any unnecessary items.
        """
        if wandb:
            wandb.finish()

    def _load_checkpoint(self, path: str) -> dict:
        """
        Load a pre-trained checkpoint of a specific name.

        Check if a checkpoint exists, and if so, load and return the checkpoint.
        If no checkpoint is found, return None.

        Parameters
        ----------
        path : str
            A ``string`` of the path to load the checkpoint from.

        Returns
        -------
        dict
            Returns a ``dict`` of the loaded model and metadata. Returns `None`
            if no checkpoint can be found.
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            return checkpoint
        else:
            return None

    def _create_test_image(self) -> None:
        """
        Load the test image to be used to verify the model after every epoch.
        """
        image = Image.open('media/waterfalls-low-res.png')
        image = ToTensor()(image)
        self.test_image = image.unsqueeze(0).to(self.device)

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
            step_size=self.epochs // 8,
            gamma=0.6
        )
        self.gen_scheduler = torch.optim.lr_scheduler.StepLR(
            self.gen_optimizer,
            step_size=self.epochs // 8,
            gamma=0.6
        )
        self.scaler = amp.GradScaler(enabled=self.amp)

    def _initialize_trainer(self) -> None:
        """
        Setup the trainer by initializing all models, loss functions,
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

    def _log_wandb(self, contents: dict, step: int = None) -> None:
        """
        Log a dictionary of contents to Weights and Biases if configured.

        Parameters
        ----------
        contents : dict
            A ``dictionary`` of information to log to Weights and Biases.
        step : int
            An ``int`` of the current step.
        """
        if wandb and self.main_process:
            wandb.log(contents, step=step)

    def _model_state(self, epoch: int, phase: str) -> dict:
        """
        Create a model save state with metadata.

        Various metadata points related to training are valuable to be included
        in the checkpoint to make it easier to pickup progress where it was
        left off while continuing from an existing checkpoint.

        Parameters
        ----------
        epoch : int
            An ``int`` of the current epoch in the training pass.
        phase : str
            A ``string`` of the current training phase.

        Returns
        -------
        dict
            Returns a ``dict`` of the latest model state including metadata
            information.
        """
        return {
            "epoch": epoch,
            "phase": phase,
            "state": self.generator.state_dict()
        }

    def _test(self, epoch: int, phase: str, step: int) -> None:
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
        phase : str
            A ``string`` of the current training phase.
        step : int
            An ``int`` of the current step in the training pass.
        """
        self.generator.eval()

        self._log(f'Testing results after epoch {epoch}')

        with torch.no_grad():
            loss, psnr = 0.0, 0.0
            start_time = time.time()

            for low_res, _, high_res in tqdm(self.test_loader, disable=not self.main_process):
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)

                super_res = self.generator(low_res).to(self.device)

                psnr += 10 * log10(1 / ((super_res - high_res) ** 2).mean().item())

                loss += self.val_loss(super_res, high_res)

            end_time = time.time()
            time_taken = end_time - start_time
            throughput = len(self.test_loader) * self.world_size / time_taken
            psnr = psnr / len(self.test_loader)
            loss = loss / len(self.test_loader)            

            self._log(f'PSNR: {round(psnr, 3)}, '
                      f'Throughput: {round(throughput, 3)} images/sec')

            # Strip so we get just the short phase, ie 'gan' or 'psnr'
            short_phase = ''.join(phase.split('-')[1:])
            self._log_wandb(
                {
                    f'{short_phase}/PSNR': psnr,
                    f'{short_phase}/val-loss': loss,
                    f'{short_phase}/throughput/test': throughput,
                    f'{short_phase}/epoch': epoch
                },
                step=step
            )

            if psnr > self.best_psnr and self.main_process:
                self.best_psnr = psnr
                torch.save(self._model_state(epoch, phase),
                           f'{phase}-best.pth')
            if self.main_process:
                torch.save(self._model_state(epoch, phase),
                           f'{phase}-latest.pth')

            # If the user requested to not save images, return immediately and
            # avoid generating and saving the image.
            if not self.save_image:
                return
            # Save a copy of a single image that has been super-resed for easy
            # tracking of progress.
            super_res = self.generator(self.test_image).to(self.device)
            utils.save_image(super_res, f'output/SR_epoch{epoch}.png', padding=5)
            _, _, height, width = super_res.shape
            super_res_resize = Resize((height // 4, width // 4),
                                       interpolation=InterpolationMode.BICUBIC)(super_res)
            if wandb:
                self._log_wandb(
                    {f'images/epoch{epoch}': wandb.Image(super_res_resize)}
                )

    def train(self) -> None:
        """
        Initiate the pre-training followed by the main training phases of the
        network.
        """
        self._pretrain()
        # Clear the GPU cache between pre-training and main training phases to
        # optimize memory usage during extended runtime.
        torch.cuda.empty_cache()
        self._gan_train()
        self._cleanup()
