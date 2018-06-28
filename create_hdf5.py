#! /usr/bin/env python2

import glob
import numpy as np
import h5py
import sys
from PIL import Image
import PIL
import random
from utils import *

def generate_hdf5():
    PATCH_SIZE = 64
    STEP = 0
    STRIDE = 128
    BATCH_SIZE = 64
    DATA_AUG_TIMES = 1
    W = 512
    H = 512
    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    
    src_dir = './data/mb2014_png'
    dst_dir = './data/mb2014_bin/'
    hdf5_path   = dst_dir + ('data_da%d_W%d_H%d_p%d_s%d_b%d.hdf5' 
            % (DATA_AUG_TIMES, W, H, PATCH_SIZE, STRIDE, BATCH_SIZE))

    fpxleft_tr  = glob.glob(src_dir + '/train/X_left/*.png')
    fpxright_tr = glob.glob(src_dir + '/train/X_right/*.png')
    fpyleft_tr  = glob.glob(src_dir + '/train/Y_left/*.png')
    fpyright_tr = glob.glob(src_dir + '/train/Y_right/*.png')

    count = 0
    for i in xrange(len(fpxleft_tr)):
        img = Image.open(fpxleft_tr[i])
        for s in xrange(len(scales)):
            newsize = (int(img.size[0]*scales[s]), int(img.size[1]*scales[s]))
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            im_h, im_w = img_s.size
            for x in range(0+STEP, (im_h-PATCH_SIZE), STRIDE):
                for y in range(0+STEP, (im_w-PATCH_SIZE), STRIDE):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES
    if origin_patch_num % BATCH_SIZE != 0:
        numPatches = (origin_patch_num/BATCH_SIZE+1) * BATCH_SIZE
    else:
        numPatches = origin_patch_num

    fpxleft_eval  = glob.glob(src_dir + '/eval/X_left/*.png')
    fpxright_eval = glob.glob(src_dir + '/eval/X_right/*.png')
    fpyleft_eval  = glob.glob(src_dir + '/eval/Y_left/*.png')
    fpyright_eval = glob.glob(src_dir + '/eval/Y_right/*.png')

    fpxleft_te  = glob.glob(src_dir + '/test/X_left/*.png')
    fpxright_te = glob.glob(src_dir + '/test/X_right/*.png')
    fpyleft_te  = glob.glob(src_dir + '/test/Y_left/*.png')
    fpyright_te = glob.glob(src_dir + '/test/Y_right/*.png')

    print("[*] Information...")
    print("\tNumber of train images %d" % len(fpxleft_tr))
    print("\tNumber of eval  images %d" % len(fpxleft_eval))
    print("\tNumber of test  images %d" % len(fpxleft_te))
    print("\tPatch size = %d" % PATCH_SIZE)
    print("\tBatch size = %d" % BATCH_SIZE)
    print("\tTotal patches = %d" % numPatches)
    print("\tTotal batches = %d" % (numPatches/BATCH_SIZE))
    print("\tDATA_AUG_TIMES = {}".format(DATA_AUG_TIMES))
    print("\tSourc dir=%s" % src_dir)
    print("\tDestination HD5 file=%s" % hdf5_path)
    print("\tAll test and eval images are resized to %dx%dx3" % (W,H))
    sys.stdout.flush()

    shape_tr   = (numPatches, PATCH_SIZE, PATCH_SIZE, 3)
    shape_eval = (len(fpxleft_eval), W, H, 3)
    shape_te   = (len(fpxleft_te  ), W, H, 3)

    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("XL_tr", shape_tr, np.uint8)
    hdf5_file.create_dataset("XR_tr", shape_tr, np.uint8)
    hdf5_file.create_dataset("YL_tr", shape_tr, np.uint8)
    hdf5_file.create_dataset("YR_tr", shape_tr, np.uint8)

    hdf5_file.create_dataset("XL_eval", shape_eval, np.uint8)
    hdf5_file.create_dataset("XR_eval", shape_eval, np.uint8)
    hdf5_file.create_dataset("YL_eval", shape_eval, np.uint8)
    hdf5_file.create_dataset("YR_eval", shape_eval, np.uint8)

    hdf5_file.create_dataset("XL_te", shape_te, np.uint8)
    hdf5_file.create_dataset("XR_te", shape_te, np.uint8)
    hdf5_file.create_dataset("YL_te", shape_te, np.uint8)
    hdf5_file.create_dataset("YR_te", shape_te, np.uint8)

    print("[*] Processing Train Images")
    c = 0
    for i in xrange(len(fpxleft_tr)):
        imgL  = Image.open(fpxleft_tr[i])
        imgR  = Image.open(fpxright_tr[i])
        imgYL = Image.open(fpyleft_tr[i])
        imgYR = Image.open(fpyright_tr[i])
 
        print("\t Tr image \#%3d" % (i+1))
        sys.stdout.flush()

        for s in xrange(len(scales)):
            newsize = (int(imgL.size[0]*scales[s]), int(imgL.size[1]*scales[s]))
            imgL_s  = imgL.resize( newsize, resample=PIL.Image.BICUBIC)
            imgR_s  = imgR.resize( newsize, resample=PIL.Image.BICUBIC)
            imgYL_s = imgYL.resize(newsize, resample=PIL.Image.BICUBIC)
            imgYR_s = imgYR.resize(newsize, resample=PIL.Image.BICUBIC)

            imgL_s  = np.reshape(np.array(imgL_s , dtype="uint8"), (imgL_s.size[0] , imgL_s.size[1] , 3))
            imgR_s  = np.reshape(np.array(imgR_s , dtype="uint8"), (imgR_s.size[0] , imgR_s.size[1] , 3))
            imgYL_s = np.reshape(np.array(imgYL_s, dtype="uint8"), (imgYL_s.size[0], imgYL_s.size[1], 3))
            imgYR_s = np.reshape(np.array(imgYR_s, dtype="uint8"), (imgYR_s.size[0], imgYR_s.size[1], 3))

            for j in xrange(DATA_AUG_TIMES):
                im_h, im_w, _ = imgL_s.shape
                assert im_h == imgR_s.shape[0]
                assert im_w == imgR_s.shape[1]
                assert im_h == imgYL_s.shape[0]
                assert im_w == imgYL_s.shape[1]
                assert im_h == imgYR_s.shape[0]
                assert im_w == imgYR_s.shape[1]
                for x in range(0+STEP, im_h-PATCH_SIZE, STRIDE):
                    for y in range(0+STEP, im_w-PATCH_SIZE, STRIDE):
                        mode = random.randint(0, 7)
                        hdf5_file["XL_tr"][c, ...] = data_augmentation(imgL_s[x:x+PATCH_SIZE , y:y+PATCH_SIZE, :], mode)
                        hdf5_file["XR_tr"][c, ...] = data_augmentation(imgR_s[x:x+PATCH_SIZE , y:y+PATCH_SIZE, :], mode)
                        hdf5_file["YL_tr"][c, ...] = data_augmentation(imgYL_s[x:x+PATCH_SIZE, y:y+PATCH_SIZE, :], mode)
                        hdf5_file["YR_tr"][c, ...] = data_augmentation(imgYR_s[x:x+PATCH_SIZE, y:y+PATCH_SIZE, :], mode)
                        c += 1

    print("[*] Processing Evaluation Images")
    c = 0
    for i in xrange(len(fpxleft_eval)):
        imgL  = Image.open(fpxleft_eval[i])
        imgR  = Image.open(fpxright_eval[i])
        imgYL = Image.open(fpyleft_eval[i])
        imgYR = Image.open(fpyright_eval[i])

        print("\t Evaluation image \#%3d" % (i+1))
        sys.stdout.flush()
        imgL_s  = imgL.resize( (W,H), resample=PIL.Image.BICUBIC)
        imgR_s  = imgR.resize( (W,H), resample=PIL.Image.BICUBIC)
        imgYL_s = imgYL.resize((W,H), resample=PIL.Image.BICUBIC)
        imgYR_s = imgYR.resize((W,H), resample=PIL.Image.BICUBIC)

        hdf5_file["XL_eval"][c, ...] = np.reshape(np.array(imgL_s , dtype="uint8"), (imgL_s.size[0] , imgL_s.size[1] , 3))
        hdf5_file["XR_eval"][c, ...] = np.reshape(np.array(imgR_s , dtype="uint8"), (imgR_s.size[0] , imgR_s.size[1] , 3))
        hdf5_file["YL_eval"][c, ...] = np.reshape(np.array(imgYL_s, dtype="uint8"), (imgYL_s.size[0], imgYL_s.size[1], 3))
        hdf5_file["YR_eval"][c, ...] = np.reshape(np.array(imgYR_s, dtype="uint8"), (imgYR_s.size[0], imgYR_s.size[1], 3))
        c += 1

    print("[*] Processing Test Images")
    c = 0
    for i in xrange(len(fpxleft_te)):
        imgL  = Image.open(fpxleft_te[i])
        imgR  = Image.open(fpxright_te[i])
        imgYL = Image.open(fpyleft_te[i])
        imgYR = Image.open(fpyright_te[i])

        print("\t Test image \#%3d" % (i+1))
        sys.stdout.flush()

        imgL_s  = imgL.resize( (W,H), resample=PIL.Image.BICUBIC)
        imgR_s  = imgR.resize( (W,H), resample=PIL.Image.BICUBIC)
        imgYL_s = imgYL.resize((W,H), resample=PIL.Image.BICUBIC)
        imgYR_s = imgYR.resize((W,H), resample=PIL.Image.BICUBIC)

        hdf5_file["XL_te"][c, ...] = np.reshape(np.array(imgL_s , dtype="uint8"), (imgL_s.size[0] , imgL_s.size[1] , 3))
        hdf5_file["XR_te"][c, ...] = np.reshape(np.array(imgR_s , dtype="uint8"), (imgR_s.size[0] , imgR_s.size[1] , 3))
        hdf5_file["YL_te"][c, ...] = np.reshape(np.array(imgYL_s, dtype="uint8"), (imgYL_s.size[0], imgYL_s.size[1], 3))
        hdf5_file["YR_te"][c, ...] = np.reshape(np.array(imgYR_s, dtype="uint8"), (imgYR_s.size[0], imgYR_s.size[1], 3))

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    hdf5_file.close()

if __name__ == '__main__':
    generate_hdf5()

