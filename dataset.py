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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (Compose,
                                    RandomCrop,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    Resize,
                                    ToPILImage,
                                    ToTensor)


SUPPORTED_IMAGES = ('.jpg', '.jpeg', '.png')


def to_image():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])


def _image_dataset(directory):
    images = [os.path.join(directory, fn) for fn in os.listdir(directory)
              if fn.lower().endswith(SUPPORTED_IMAGES)]
    return images


class TrainData(Dataset):
    def __init__(self, dataset, crop_size, upscale_factor):
        super(TrainData, self).__init__()
        self.images = _image_dataset(dataset)

        self.lr_transform = Compose([
            ToPILImage(),
            Resize((crop_size // upscale_factor, crop_size // upscale_factor), interpolation=Image.BICUBIC),
            ToTensor()
        ])
        self.hr_transform = Compose([
            RandomCrop((crop_size, crop_size)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index])

        high_res = self.hr_transform(image)
        low_res = self.lr_transform(high_res)
        return low_res, high_res

    def __len__(self):
        return len(self.images)


class TestData(Dataset):
    def __init__(self, dataset, crop_size=96, upscale_factor=4):
        super(TestData, self).__init__()
        self.upscale_factor = upscale_factor
        self.images = _image_dataset(dataset)

        self.lr_transform = Compose([
            ToPILImage(),
            Resize((crop_size // upscale_factor, crop_size // upscale_factor), interpolation=Image.BICUBIC),
            ToTensor()
        ])
        self.bicubic = Compose([
            ToPILImage(),
            Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
            ToTensor()
        ])
        self.hr_transform = Compose([
            RandomCrop((crop_size, crop_size)),
            ToTensor()
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index])

        high_res = self.hr_transform(image)
        low_res = self.lr_transform(high_res)
        bicubic = self.bicubic(low_res)
        return low_res, bicubic, high_res

    def __len__(self):
        return len(self.images)


def train_dataset(train_directory, batch_size, crop_size=96, upscale_factor=4):
    train_data = TrainData(train_directory,
                           crop_size=crop_size,
                           upscale_factor=upscale_factor)
    trainloader = DataLoader(
        dataset=train_data,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True
    )
    return trainloader


def test_dataset(test_directory, upscale_factor=4):
    test_data = TestData(test_directory, upscale_factor=upscale_factor)
    testloader = DataLoader(
        dataset=test_data,
        num_workers=1,
        batch_size=1,
        shuffle=False
    )
    return testloader
