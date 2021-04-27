import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomCrop,
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
        self.preprocess_high_res = Compose([CenterCrop(384),
                                            RandomCrop(crop_size),
                                            ToTensor()])
        self.preprocess_low_res = Compose([ToPILImage(),
                                           Resize(crop_size // upscale_factor,
                                                  interpolation=Image.BICUBIC),
                                           ToTensor()])

    def __getitem__(self, index):
        high_res = self.preprocess_high_res(Image.open(self.images[index]))
        low_res = self.preprocess_low_res(high_res)
        return low_res, high_res

    def __len__(self):
        return len(self.images)


class ValData(Dataset):
    def __init__(self, dataset, upscale_factor):
        super(ValData, self).__init__()
        self.upscale_factor = upscale_factor
        self.images = image_dataset(dataset)

    def __getitem__(self, index):
        high_res = Image.open(self.images[index])
        crop_size = 128
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        high_res = CenterCrop(crop_size)(high_res)
        low_res = lr_scale(high_res)
        high_res_restore = hr_scale(low_res)
        norm = ToTensor()
        return norm(low_res), norm(high_res_restore), norm(high_res)

    def __len__(self):
        return len(self.images)
