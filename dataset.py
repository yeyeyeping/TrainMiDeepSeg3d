import itertools
import random

import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import interpolation
from scipy.ndimage import binary_erosion, binary_dilation
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils import *


def nifty2array(path):
    img_itk = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img_itk)
    return data


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


class BraTS2018(Dataset):
    """ BraTS2018 Dataset """

    def __init__(self, base_dir=None, split='train', full_num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir + '/train.txt', 'r') as f:
                self.sample_list = f.readlines()
        else:
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = f.readlines()
        self.sample_list = [item.strip() for item in self.sample_list]
        if full_num is not None:
            self.sample_list = self.sample_list[:full_num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def crop_volume(self, data, start_end_points):
        minZidx = start_end_points[0]
        maxZidx = start_end_points[1]
        minXidx = start_end_points[2]
        maxXidx = start_end_points[3]
        minYidx = start_end_points[4]
        maxYidx = start_end_points[5]
        croped_volume = data[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
        return croped_volume

    def __getitem__(self, idx):
        image_name = self.sample_list[idx]
        img_path = self._base_dir + "/../processed_brats18/{}_img.nii.gz".format(image_name)
        lab_path = self._base_dir + "/../processed_brats18/{}_lab.nii.gz".format(image_name)
        image = nifty2array(img_path)
        label = nifty2array(lab_path)
        image = itensity_normalize_one_volume(image)
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)

        bbox, points = self.create_bbox(sample["label"])
        sample["points"] = points

        select_seeds = self.rand_select_point(sample["label"])
        self.seed2map(select_seeds, sample["points"])
        self.crop_volume(sample, bbox)

        self.preprocessing_input(sample)
        # self.label_onehot(sample)
        del sample["points"]
        return sample

    def preprocessing_input(self, sample):
        img, points = sample["image"], sample["points"]
        geodesic = interaction_geodesic_distance(img, points)

        zoomed_img = zoom_volume_size(img)
        zoomed_dis = zoom_volume_size(geodesic)
        sample["label"] = torch.from_numpy(zoom_volume_size(sample["label"])).long()

        sample["image"] = torch.from_numpy(np.stack([zoomed_img, zoomed_dis]))

    def label_onehot(self, sample):
        *_, slices, h, w = sample["label"].shape
        mask = torch.zeros(size=[2, slices, h, w])
        sample["label"] = mask.scatter_(0,
                                        torch.from_numpy(sample["label"]).unsqueeze(0).long(),
                                        1)

    def seed2map(self, seeds, points):
        for (z, y, x) in seeds:
            points[z, y, x] = 1

    def rand_select_point(self, mask):
        kernel = np.ones((5, 5, 5), np.uint8)
        # 对mask腐蚀在膨胀
        erode = binary_erosion(mask, structure=kernel, iterations=1)
        dilate = binary_dilation(mask, structure=kernel, iterations=1)
        # 在腐蚀和膨胀中间部分抽样用以模拟用户无法精准点击extreme point
        dilate[erode == 1] = 0

        k = random.randint(0, 5)
        z, y, x = np.where(dilate)
        rand_idx = random.choices(range(len(x)), k=k)
        rand_points = list(zip(z[rand_idx], y[rand_idx], x[rand_idx]))
        return rand_points

    def crop_volume(self, data, start_end_points):
        minZidx = start_end_points[0]
        maxZidx = start_end_points[1]
        minXidx = start_end_points[2]
        maxXidx = start_end_points[3]
        minYidx = start_end_points[4]
        maxYidx = start_end_points[5]
        data["image"] = data["image"][minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
        data["label"] = data["label"][minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
        data["points"] = data["points"][minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]

    def create_bbox(self, mask, pert=10):
        z, y, x = np.where(mask != 0)
        point = np.zeros_like(mask, dtype=np.uint8)
        # 生成极值点，并增加扰动
        maxzidx, minzidx = min(mask.shape[0], np.argmax(z) + random.randint(-pert, pert)), max(0, np.argmin(
            z) + random.randint(-pert, pert))
        point[z[maxzidx], y[maxzidx], x[maxzidx]] = 1
        point[z[minzidx], y[minzidx], x[minzidx]] = 1

        maxyidx, minyidx = min(mask.shape[1], np.argmax(y) + random.randint(-pert, pert)), max(0, np.argmin(
            y) + random.randint(-pert, pert))
        point[z[maxyidx], y[maxyidx], x[maxyidx]] = 1
        point[z[minyidx], y[minyidx], x[minyidx]] = 1

        maxxidx, minxidx = min(mask.shape[2], np.argmax(x) + random.randint(-pert, pert)), max(0, np.argmin(
            x) + random.randint(-pert, pert))
        point[z[maxxidx], y[maxxidx], x[maxxidx]] = 1
        point[z[minxidx], y[minxidx], x[minxidx]] = 1

        return [
            max(np.min(z) - pert, 0), min(np.max(z) + pert, mask.shape[0]),
            max(np.min(y) - pert, 0), min(np.max(y) + pert, mask.shape[1]),
            max(np.min(x) - pert, 0), min(np.max(x) + pert, mask.shape[2])
        ], point


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class ReScale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        spacingzxy = [self.output_size[0] / w,
                      self.output_size[1] / h, self.output_size[2] / d]

        image = interpolation.zoom(image, spacingzxy, order=3)
        label = interpolation.zoom(label, spacingzxy, order=0)
        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2], image.shape[3]).float()
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': image, 'label': sample['label'].long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    from unet3d import Unet3d
    from loss import DiceLoss

    net = Unet3d(in_dim=2, out_dim=2, num_filter=8)
    loss = DiceLoss()
    folder = "processed_brats18"
    dt = BraTS2018(base_dir=folder)
    item = dt.__getitem__(1)
    out = torch.softmax(net(item["image"].unsqueeze(0)), dim=1)
    loss = loss(out, item["label"].unsqueeze(0))
    print(loss)
