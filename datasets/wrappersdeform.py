import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, make_coord
import torch.nn.functional as F

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

@register('realsrarbi-deform-paired')
class RealSRPaired(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        scale = random.choice([1.5, 2, 2.5, 3, 3.5, 4])

        if scale == 1.5:
            img_lr = self.dataset.dataset_lr1dot5[idx]
            img_hr = self.dataset.dataset_hr1dot5[idx]
        if scale == 2:
            img_lr = self.dataset.dataset_lr2[idx]
            img_hr = self.dataset.dataset_hr2[idx]
        if scale == 2.5:
            img_lr = self.dataset.dataset_lr2dot5[idx]
            img_hr = self.dataset.dataset_hr2dot5[idx]
        if scale == 3:
            img_lr = self.dataset.dataset_lr3[idx]
            img_hr = self.dataset.dataset_hr3[idx]
        if scale == 3.5:
            img_lr = self.dataset.dataset_lr3dot5[idx]
            img_hr = self.dataset.dataset_hr3dot5[idx]
        if scale == 4:
            img_lr = self.dataset.dataset_lr4[idx]
            img_hr = self.dataset.dataset_hr4[idx]

        w = self.inp_size

        x0 = random.randrange(0, img_lr.shape[-2] - w, 2)
        y0 = random.randrange(0, img_lr.shape[-1] - w, 2)
        crop_lr = img_lr[:, x0: x0 + w, y0: y0 + w]
        crop_hr = img_hr[:, int(x0 * scale): int(x0 * scale + w * scale),
                  int(y0 * scale): int(y0 * scale + w * scale)]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        crop_lr_resized = resize_fn(crop_lr, (crop_hr.shape[-2], crop_hr.shape[-1]))

        diff = crop_hr - crop_lr_resized

        diff_coord, diff_rgb = to_pixel_samples(diff.contiguous())

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            diff_coord = diff_coord[sample_lst]
            diff_rgb = diff_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'diff_rgb': diff_rgb
        }


@register('realsrarbi-test-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        crop_lr, crop_hr = img_lr, img_hr

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
