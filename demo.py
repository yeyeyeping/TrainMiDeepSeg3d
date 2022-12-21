# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
import os
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib.pyplot as plt

'''
绿色是浮肿区域(ED),标签为2;

黄色为增强肿瘤区域(ET)标签是4;

红色为坏疽(Net)标签是1;

背景标签是0.

WT = ED+ET+Net 即 WT = 2+4+1

TC = ET+Net 即TC = 4+1

ET = 4

'''


def testBoundingBox():
    img = sitk.GetArrayFromImage(sitk.ReadImage("Brats18_TCIA02_491_1_seg.nii"))
    z, y, x = np.where(img)
    points = [min(z), max(z), min(y), max(y), min(x), max(x)]
    img = crop_volume(img, points)
    sitk.WriteImage(sitk.GetImageFromArray(img), "1.nii")


def extract_whole_tuma(imgpath):
    img = sitk.GetArrayFromImage(sitk.ReadImage(imgpath))
    mask = (img != 0)
    return mask


def extract_tuma_core(imgpath):
    img = sitk.GetArrayFromImage(sitk.ReadImage(imgpath))
    mask = ((img == 4) | (img == 1))
    return mask


def preprocess(folder):
    pths = os.listdir(folder)


import dataset

folder = "processed_brats18"
# # testBoundingBox()
# m=extract_tuma_core("Brats18_TCIA02_491_1_seg.nii")
# import  dataset
dt = dataset.BraTS2018(base_dir=folder)
item = dt.__getitem__(1)

for i in range(3, 32, 3):
    plt.subplot(3, 10, i // 3)
    plt.imshow(item["image"][0][i], cmap="gray")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xticks([])
    plt.yticks([])
for i in range(0, 32, 3):
    plt.subplot(3, 10, i // 3 + 10)
    plt.imshow(item["image"][1][i], cmap="gray")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xticks([])
    plt.yticks([])
for i in range(0, 32, 3):
    plt.subplot(3, 10, i // 3 + 20)
    plt.imshow(item["label"][1][i], cmap="gray")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xticks([])
    plt.yticks([])
plt.show()
