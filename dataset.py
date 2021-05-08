import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
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


def image_dataset(directory):
    images = [os.path.join(directory, fn) for fn in os.listdir(directory)
              if fn.lower().endswith(SUPPORTED_IMAGES)]
    return images


class TrainData(Dataset):
    def __init__(self, dataset, crop_size, upscale_factor):
        super(TrainData, self).__init__()
        self.images = image_dataset(dataset)

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


class ValData(Dataset):
    def __init__(self, dataset, crop_size=96, upscale_factor=4):
        super(ValData, self).__init__()
        self.upscale_factor = upscale_factor
        self.images = image_dataset(dataset)

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
