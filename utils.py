#! /usr/bin/env python2

import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
import PIL

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 3:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 4:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 5:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)
    elif mode == 6:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 7:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)


#class train_data():
#    def __init__(self, fxl='./data/mb2014_bin/imgL.npy',
#                       fxr='./data/mb2014_bin/imgR.npy',
#                       fyl='./data/mb2014_bin/gtL.npy',
#                       fyr='./data/mb2014_bin/gtR.npy'):
#        self.fxl = fxl
#        self.fxr = fxr
#        self.fyl = fyl
#        self.fyr = fyr
#
#        assert '.npy' in fxl
#        assert '.npy' in fxr
#        assert '.npy' in fyl
#        assert '.npy' in fyr
#
#        if not os.path.exists(fxl) or not os.path.exists(fxr) or not os.path.exists(fyl):
##           not os.path.exists(fyr):
#            print("[!] Data file not exists")
#            sys.exit(1)
#
#    def __enter__(self):
#        print("[*] Loading data...")
#        self.XL = np.load(self.fxl)
#        self.XR = np.load(self.fxr)
#        self.YL = np.load(self.fyl)
##        self.YR = np.load(self.fyr)
#        #np.random.shuffle(self.data)
#        print("[*] Load successfully...")
#        return self.XL, self.XR, self.YL
#
#    def __exit__(self, type, value, trace):
#        del self.XL
#        del self.XR
#        del self.YL
##        del self.YR
#        gc.collect()
#        print("In __exit__()")


#def load_data(fxl='./data/mb2014_bin/imL.npy',
#              fxr='./data/mb2014_bin/imR.npy',
#              fyl='./data/mb2014_bin/YL.npy',
#              fyr='./data/mb2014_bin/YR.npy'):
#    return train_data(fxl=fxl, fxr=fxr, fyl=fyl, fyr=fyr)


#def load_images(filelist, sH=1, sW=1):
#    # pixel value range 0-255
#    if not isinstance(filelist, list):
#        im = Image.open(filelist)
#        newsize = (int(im.size[0]*sH), int(im.size[1]*sW))
#        im_s = im.resize(newsize, resample=PIL.Image.BICUBIC)
#        return np.array(im_s).reshape(1, im_s.size[1], im_s.size[0], 3)
#    data = []
#    for file in filelist:
#        im = Image.open(file)
#        newsize = (int(im.size[0]*sH), int(im.size[1]*sW))
#        im_s = im.resize(newsize, resample=PIL.Image.BICUBIC)
#        data.append(np.array(im_s).reshape(1, im_s.size[1], im_s.size[0], 3))
#    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'png')


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


