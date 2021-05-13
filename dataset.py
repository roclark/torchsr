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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (Compose,
                                    RandomCrop,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    Resize,
                                    ToPILImage,
                                    ToTensor)
from torchvision.transforms.functional import InterpolationMode
from typing import Tuple


SUPPORTED_IMAGES = ('.jpg', '.jpeg', '.png')


def _image_dataset(directory: str) -> list:
    """
    Build list of images in the dataset directory.

    Given a directory, find all supported image types in the directory and save
    the path and filename for each image to be used for the dataset.

    Parameters
    ----------
    directory : str
        A ``string`` of the directory containing images to sample.

    Returns
    -------
    list
        Returns a ``list`` of image paths and names within the requested
        directory.
    """
    images = [os.path.join(directory, fn) for fn in os.listdir(directory)
              if fn.lower().endswith(SUPPORTED_IMAGES)]
    return images


class TrainData(Dataset):
    """
    Build the training dataset with transformations.

    Helper class which first creates a list of supported images given a
    directory as input and generates transformations based on those images. The
    high resolution image is taken directly from the source image and a random
    NxN cropping is taken from each image to be used for training. A low
    resolution image is generated based on the high resolution image by
    downscaling the image using bicubic interpolation. The amount the image is
    downscaled is determined by the `upscale_factor` parameter which indicates
    how much the image should be scaled down in each direction.

    Parameters
    ----------
    dataset : str
        A ``string`` of the path to the directory containing images to be used
        for training.
    crop_size : int
        An ``int`` of the size to crop the high resolution images to in pixels.
        The size is used for both the height and width, ie. a crop_size of `96`
        will take a 96x96 section of the input image.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.
    """
    def __init__(self, dataset: str, crop_size: int, upscale_factor: int) -> None:
        super(TrainData, self).__init__()
        self.images = _image_dataset(dataset)

        self.lr_transform = Compose([
            ToPILImage(),
            Resize((crop_size // upscale_factor, crop_size // upscale_factor),
                   interpolation=InterpolationMode.BICUBIC),
            ToTensor()
        ])
        self.hr_transform = Compose([
            RandomCrop((crop_size, crop_size)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Sample a single image from the dataset.

        Given an index, open the resulting image from the dataset and perform
        the high and low resolution transformations on the image before
        returning the resulting low and high resolution images.

        Parameters
        ----------
        index : int
            Returns an ``int`` of the index representing an image to pull from
            the dataset.

        Returns
        -------
        tuple
            Returns a ``tuple`` of the transformed low and high resolution
            images, respectively.
        """
        image = Image.open(self.images[index])

        high_res = self.hr_transform(image)
        low_res = self.lr_transform(high_res)
        return low_res, high_res

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns
        -------
        int
            Returns an ``int`` of the number of images in the dataset.
        """
        return len(self.images)


class TestData(Dataset):
    """
    Build the test dataset with transformations.

    Helper class which first creates a list of supported images given a
    directory as input and generates transformations based on those images. The
    high resolution image is taken directly from the source image and a random
    NxN cropping is taken from each image to be used for testing. A low
    resolution image is generated based on the high resolution image by
    downscaling the image using bicubic interpolation. The amount the image is
    downscaled is determined by the `upscale_factor` parameter which indicates
    how much the image should be scaled down in each direction. For
    comparisons, a bicubic image is also created based on the low resolution
    image to see if generated images are better than a simple bicubic upsample.

    Parameters
    ----------
    dataset : str
        A ``string`` of the path to the directory containing images to be used
        for testing.
    crop_size : int
        An ``int`` of the size to crop the high resolution images to in pixels.
        The size is used for both the height and width, ie. a crop_size of `96`
        will take a 96x96 section of the input image.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.
    """
    def __init__(self, dataset: str, crop_size: int = 96, upscale_factor: int = 4) -> None:
        super(TestData, self).__init__()
        self.upscale_factor = upscale_factor
        self.images = _image_dataset(dataset)

        self.lr_transform = Compose([
            ToPILImage(),
            Resize((crop_size // upscale_factor, crop_size // upscale_factor),
                   interpolation=InterpolationMode.BICUBIC),
            ToTensor()
        ])
        self.bicubic = Compose([
            ToPILImage(),
            Resize((crop_size, crop_size),
                   interpolation=InterpolationMode.BICUBIC),
            ToTensor()
        ])
        self.hr_transform = Compose([
            RandomCrop((crop_size, crop_size)),
            ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample a single image from the dataset.

        Given an index, open the resulting image from the dataset and perform
        the high and low resolution transformations on the image before
        returning the resulting low and high resolution images.

        Parameters
        ----------
        index : int
            Returns an ``int`` of the index representing an image to pull from
            the dataset.

        Returns
        -------
        tuple
            Returns a ``tuple`` of the transformed low, bicubic, and high
            resolution images, respectively.
        """
        image = Image.open(self.images[index])

        high_res = self.hr_transform(image)
        low_res = self.lr_transform(high_res)
        bicubic = self.bicubic(low_res)
        return low_res, bicubic, high_res

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns
        -------
        int
            Returns an ``int`` of the number of images in the dataset.
        """
        return len(self.images)


def train_dataset(train_directory: str, batch_size: int, crop_size: int = 96,
                  upscale_factor: int = 4) -> DataLoader:
    """
    Build a training dataset based on the input directory.

    Create a PyTorch DataLoader instance with multiple pipelined workers to
    reduce data reading and processing bottlenecks.

    Parameters
    ----------
    train_directory : str
        A ``string`` of the directory containing supported images.
    batch_size : int
        An ``int`` of the number of images to include in each batch during
        training.
    crop_size : int
        An ``int`` of the size to crop the high resolution images to in pixels.
        The size is used for both the height and width, ie. a crop_size of `96`
        will take a 96x96 section of the input image.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.

    Returns
    -------
    DataLoader
        Returns a ``DataLoader`` instance of the training dataset.
    """
    train_data = TrainData(train_directory,
                           crop_size=crop_size,
                           upscale_factor=upscale_factor)
    trainloader = DataLoader(
        dataset=train_data,
        num_workers=16,
        batch_size=batch_size,
        shuffle=True
    )
    return trainloader


def test_dataset(test_directory: str,
                 upscale_factor: int = 4) -> DataLoader:
    """
    Build a testing dataset based on the input directory.

    Create a PyTorch DataLoader instance to be used to test the latest model
    after each epoch.

    Parameters
    ----------
    test_directory : str
        A ``string`` of the directory containing supported images.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.

    Returns
    -------
    DataLoader
        Returns a ``DataLoader`` instance of the testing dataset.
    """
    test_data = TestData(test_directory, upscale_factor=upscale_factor)
    testloader = DataLoader(
        dataset=test_data,
        num_workers=1,
        batch_size=1,
        shuffle=False
    )
    return testloader
