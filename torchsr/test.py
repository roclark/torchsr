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
from argparse import Namespace
from collections import OrderedDict
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from typing import NoReturn


def test(args: Namespace, model: object, device: str) -> NoReturn:
    """
    Generate an upreset image.

    Generate a super resolution image based on the latest trained model and
    save the output to a new file.

    Parameters
    ----------
    args : Namespace
        A ``Namespace`` containing all of the arguments passed by the user
        including defaults.
    model : object
        The Generator class declaration for the specified model.
    device : str
        A ``string`` of the primary device to use for computation, such as
        `cuda` for NVIDIA GPUs.
    """
    generator = model().to(device)
    state_dict = torch.load(f'{args.model.lower()}-gan-best.pth')

    new_state_dict = OrderedDict()

    # Remove the 'module.' prefix from all keys to make it work regardless of
    # the number of compute devices that were used for training. Otherwise, if
    # the model was trained using distributed mode, the testing module will be
    # confused as it looks for the 'module.' prefix in all weights which is
    # common for models trained with DataParallel or DistributedDataParallel.
    for key, value in state_dict.items():
        if key.startswith('module.'):
            name = key[len('module.'):]
        new_state_dict[name] = value

    generator.load_state_dict(new_state_dict)

    image = Image.open(args.image)
    low_res = ToTensor()(image)
    low_res = low_res.unsqueeze(0)
    low_res = low_res.to(device)

    super_res = generator(low_res)
    save_image(super_res, f'upres-{args.image}')
