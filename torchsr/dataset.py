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
import cv2
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import (Compose,
                                    RandomCrop,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    Resize,
                                    ToPILImage,
                                    ToTensor)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from typing import Tuple


MAX_CROP = 128  # Maximum size that images will be cropped to for training
SUBDIVISION_DIMS = 480  # Dimensions to attempt to crop images to
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


def _save_image(image: np.array, image_name: str, row: int,
                column: int, height: int, width: int) -> None:
    """
    Save a subdivided image to the local filesystem.

    Given an image and dimensions to start a crop at for the image, determine
    if the image subset is greater than or equal to the MAX_CROP constant to
    ensure that the subdivided image can be used for all models which take a
    random sample from each sub-image. If the subsection is smaller, take the
    final MAX_CROP pixels for the necessary dimension(s) to represent the
    subdivided image. This will have overlap in some of the training images,
    but better to have minimal overlap than less diversity.

    Parameters
    ----------
    image : numpy array
        A ``numpy array`` representing the image to subdivide.
    image_name : str
        A ``string`` of the name to save the new subdivided image as.
    row : int
        An ``int`` of the row to begin taking the sample from.
    column : int
        An ``int`` of the column to begin taking the sample from.
    height : int
        An ``int`` of the overall height of the input image in pixels.
    width : int
        An ``int`` of the overall width of the input image in pixels.
    """
    # Row is too narrow - need to overlap with previous section if possible.
    if height - (row + SUBDIVISION_DIMS) < MAX_CROP:
        if height - MAX_CROP < 0:
            print('Requested image is too narrow to be subdivided. Skipping.')
            return
        row = max(height - SUBDIVISION_DIMS, 0)
    # Column is too narrow - need to overlap with previous section if possible.
    if width - (column + SUBDIVISION_DIMS) < MAX_CROP:
        if width - MAX_CROP < 0:
            print('Requested image is too short to be subdivided. Skipping.')
            return
        column = max(width - SUBDIVISION_DIMS, 0)

    cv2.imwrite(image_name,
        image[
            row: row + SUBDIVISION_DIMS,
            column: column + SUBDIVISION_DIMS,
            :
        ]
    )


def subdivide_images(directory: str, out_dir: str) -> None:
    """
    Subdivide images into smaller chunks to sample from.

    To ensure samples from the images are taken from unique locations in each
    image, each image should be subdivided into 480x480 subsets (if possible).
    To capture the actual dataset, a random cropping within each subset will be
    sampled to minimize overlapping images in the dataset.

    The subdivided images will be saved to a new directory which should then be
    used for the new training dataset for future runs.

    Parameters
    ----------
    directory : str
        A ``string`` of the directory containing images to sample.
    out_dir : str
        A ``string`` of the directory to save subdivided images to.
    """
    print('Subdividing training dataset into smaller croppings')
    print('Images will be saved to the specified output directory')
    print('This may take some time...')

    images = _image_dataset(directory)
    os.makedirs(out_dir, exist_ok=True)

    for image_name in tqdm(images):
        image = cv2.imread(image_name)

        if image.ndim == 2:
            height, width = image.shape
        elif image.ndim == 3:
            height, width, _ = image.shape

        filepath, extension = os.path.splitext(image_name)
        filename = os.path.basename(filepath)
        new_path = os.path.join(out_dir, filename)
        sub_image = 0

        for row in range(0, height, SUBDIVISION_DIMS):
            for column in range(0, width, SUBDIVISION_DIMS):
                sub_image_name = f'{new_path}_{sub_image}{extension}'
                sub_image += 1
                _save_image(image, sub_image_name, row, column, height, width)


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
    dataset : list
        A ``list`` of the full paths of images to be used for training.
    crop_size : int
        An ``int`` of the size to crop the high resolution images to in pixels.
        The size is used for both the height and width, ie. a crop_size of `96`
        will take a 96x96 section of the input image.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.
    dataset_multiplier : int
        An ``int`` of the amount to augment the dataset to increase the number
        of samples per image.
    """
    def __init__(self, dataset: str, crop_size: int, upscale_factor: int,
                 dataset_multiplier: int) -> None:
        super(TrainData, self).__init__()
        self.images = dataset * dataset_multiplier

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
        height, width = image.size
        if height < MAX_CROP or width < MAX_CROP:
            print(self.images[index])

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
    dataset : list
        A ``list`` of the full paths of images to be used for testing.
    crop_size : int
        An ``int`` of the size to crop the high resolution images to in pixels.
        The size is used for both the height and width, ie. a crop_size of `96`
        will take a 96x96 section of the input image.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.
    """
    def __init__(self, dataset: list, crop_size: int = 96, upscale_factor: int = 4) -> None:
        super(TestData, self).__init__()
        self.upscale_factor = upscale_factor
        self.images = dataset

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


def _train_dataset(train_subset: list, batch_size: int, crop_size: int = 96,
                   upscale_factor: int = 4,
                   dataset_multiplier: int = 1,
                   workers: int = 16,
                   distributed: bool = False) -> DataLoader:
    """
    Build a training dataset based on the input directory.

    Create a PyTorch DataLoader instance with multiple pipelined workers to
    reduce data reading and processing bottlenecks.

    Parameters
    ----------
    train_subset : list
        A ``list`` of images to be used in the training dataset.
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
    dataset_multiplier : int
        An ``int`` of the amount to augment the dataset to increase the number
        of samples per image.
    workers : int
        An ``int`` of the number of workers to use for loading and
        preprocessing images.
    distributed : bool
        A ``boolean`` which evaluates to `True` if the application should be
        run in distributed mode.

    Returns
    -------
    DataLoader
        Returns a ``DataLoader`` instance of the training dataset.
    """
    train_data = TrainData(train_subset,
                           crop_size=crop_size,
                           upscale_factor=upscale_factor,
                           dataset_multiplier=dataset_multiplier)
    if distributed:
        train_sampler = DistributedSampler(train_data)
        trainloader = DataLoader(
            dataset=train_data,
            sampler=train_sampler,
            num_workers=workers,
            batch_size=batch_size,
            persistent_workers=True
        )
    else:
        trainloader = DataLoader(
            dataset=train_data,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=True
        )
    return trainloader


def _test_dataset(test_subset: list,
                  upscale_factor: int = 4,
                  crop_size: int = 96,
                  workers: int = 16,
                  distributed: bool = False) -> DataLoader:
    """
    Build a testing dataset based on the input directory.

    Create a PyTorch DataLoader instance to be used to test the latest model
    after each epoch.

    Parameters
    ----------
    test_subset : list
        A ``list`` of images to be used in the testing dataset.
    upscale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.
    crop_size : int
        An ``int`` of the size to crop the high resolution images to in pixels.
        The size is used for both the height and width, ie. a crop_size of `96`
        will take a 96x96 section of the input image.
    workers : int
        An ``int`` of the number of workers to use for loading and
        preprocessing images.
    distributed : bool
        A ``boolean`` which evaluates to `True` if the application should be
        run in distributed mode.

    Returns
    -------
    DataLoader
        Returns a ``DataLoader`` instance of the testing dataset.
    """
    test_data = TestData(test_subset, crop_size=crop_size,
                         upscale_factor=upscale_factor)
    if distributed:
        test_sampler = DistributedSampler(test_data)
        testloader = DataLoader(
            dataset=test_data,
            sampler=test_sampler,
            num_workers=workers,
            batch_size=1,
            persistent_workers=True
        )
    else:
        testloader = DataLoader(
            dataset=test_data,
            num_workers=workers,
            batch_size=1,
            shuffle=False
        )
    return testloader


def initialize_datasets(train_directory: str, batch_size: int,
                        crop_size: int = 96, upscale_factor: int = 4,
                        dataset_multiplier: int = 1,
                        workers: int = 16,
                        distributed: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Initialize testing and training datasets.

    Given a directory full of high quality sample images, randomly divide the
    directory up so 90% of the images are in the training dataset and the
    remaining 10% are in the testing dataset. The datasets are then converted
    to PyTorch DataLoaders which are used to efficiently iterate through the
    images.

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
    dataset_multiplier : int
        An ``int`` of the amount to augment the dataset to increase the number
        of samples per image.
    workers : int
        An ``int`` of the number of workers to use for loading and
        preprocessing images.
    distributed : bool
        A ``boolean`` which evaluates to `True` if the application should be
        run in distributed mode.

    Returns
    -------
    Tuple
        Returns a ``tuple`` comprised of the training and testing DataLoaders,
        respectively followed by the size of each dataset.
    """
    dataset = _image_dataset(train_directory)
    train_data, test_data = train_test_split(dataset, test_size=0.1,
                                             shuffle=True)
    trainloader = _train_dataset(train_data, batch_size=batch_size,
                                 crop_size=crop_size,
                                 upscale_factor=upscale_factor,
                                 dataset_multiplier=dataset_multiplier,
                                 workers=workers,
                                 distributed=distributed)
    testloader = _test_dataset(test_data, upscale_factor=upscale_factor,
                               crop_size=crop_size, workers=workers,
                               distributed=distributed)
    return trainloader, testloader, len(train_data), len(test_data)
