# -*- coding: utf-8 -*-
from scipy import ndimage
import GeodisTK
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def geodesic_distance_3d(I, S, lamb, iter):
    '''
    get 3d geodesic disntance by raser scanning.
    I: input image
    S: binary image where non-zero pixels are used as seeds
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, [1.0, 1.0, 1.0], lamb, iter)


def resize_3D_volume_to_given_shape(volume, out_shape, order=3):
    shape0 = volume.shape
    scale_d = (out_shape[0] + 0.0) / shape0[0]
    scale_h = (out_shape[1] + 0.0) / shape0[1]
    scale_w = (out_shape[2] + 0.0) / shape0[2]
    return ndimage.interpolation.zoom(volume, [scale_d, scale_h, scale_w], order=order)


def zoom_volume_size(data):
    z, x, y = data.shape
    zoomed_volume = zoom(data, (32 / z, 64 / x, 64 / y), output=None, order=0, mode='constant', cval=0.0,
                         prefilter=True)
    return zoomed_volume


def interaction_geodesic_distance(img, seeds, threshold=0):
    """
    the geodesic distance of the extreme points
    """
    if (seeds.sum() > 0):
        reshape_image = ndimage.zoom(img, 0.5, order=1)
        reshape_point = ndimage.zoom(seeds, 0.5, order=0)
        geo_dis = geodesic_distance_3d(reshape_image, reshape_point, 1.0, 2)
        exp_dis = np.exp(-(geo_dis ** 2))
        exp_dis = resize_3D_volume_to_given_shape(exp_dis, img.shape, 1)
        lab1 = nib.Nifti1Image(exp_dis, np.eye(4))
        nib.save(lab1, "map.nii.gz")
        if threshold == 0:
            dis = exp_dis  # recale to 0-1
        if threshold > 0:
            exp_dis[exp_dis > threshold] = 1
            dis = exp_dis
    else:
        dis = np.ones_like(img, np.float)
    return dis

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)