#! /usr/bin/env python2

import argparse
import glob
from PIL import Image
import PIL
import random
from utils import *
import sys

sys.stdout.flush()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/mb2014_png')
parser.add_argument('--dst_dir', dest='save_dir', default='./data/mb2014_bin')
parser.add_argument('--data_aug_times', dest='DATA_AUG_TIMES', default=1, type=int)
parser.add_argument('--patch_size', dest='pat_size', default=32, type=int)
parser.add_argument('--step', dest='step', default=0, type=int)
parser.add_argument('--stride', dest='stride', default=10, type=int)
parser.add_argument('--batch_size', dest='bat_size', default=128, type=int)

args = parser.parse_args()

def generate_patches():
    count = 0
    fpxleft  = glob.glob(args.src_dir + '/X_left/*.png')
    fpxright = glob.glob(args.src_dir + '/X_right/*.png')
    fpyleft  = glob.glob(args.src_dir + '/Y_left/*.png')
#    fpyright = glob.glob(args.src_dir + '/Y_right/*.png')

    print("number of image pairs %d" % (len(fpxleft)-185))

    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]

    for i in xrange(len(fpxleft)-185):
        img = Image.open(fpxleft[i])
        for s in xrange(len(scales)):
            newsize = (int(img.size[0]*scales[s]), int(img.size[1]*scales[s]))
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            im_h, im_w = img_s.size
            for x in range(0+args.step, (im_h-args.pat_size), args.stride):
                for y in range(0+args.step, (im_w-args.pat_size), args.stride):
                    count += 1
    origin_patch_num = count * args.DATA_AUG_TIMES
    print("Number of extracted patches {}".format(origin_patch_num))

    if origin_patch_num % args.bat_size != 0:
        numPatches = (origin_patch_num/args.bat_size+1) * args.bat_size
    else:
        numPatches = origin_patch_num

    print("total patches = %d, batch_size = %d, total_batches = %d" %
            (numPatches, args.bat_size, numPatches/args.bat_size))

    inputsL = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype = "uint8")
    inputsR = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype = "uint8")
    inputsYL = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype = "uint8")
#    inputsYR = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype = "uint8")

    count = 0
    for i in xrange(len(fpxleft)-185):
        imgL = Image.open(fpxleft[i])
        imgR = Image.open(fpxright[i])
        imgYL = Image.open(fpyleft[i])
#        imgYR = Image.open(fpyright[i])

        for s in xrange(len(scales)):
            newsize = (int(imgL.size[0]*scales[s]), int(imgL.size[1]*scales[s]))
            imgL_s = imgL.resize(newsize, resample=PIL.Image.BICUBIC)
            imgR_s = imgR.resize(newsize, resample=PIL.Image.BICUBIC)
            imgYL_s = imgYL.resize(newsize, resample=PIL.Image.BICUBIC)
#            imgYR_s = imgYR.resize(newsize, resample=PIL.Image.BICUBIC)

            imgL_s = np.reshape(np.array(imgL_s, dtype="uint8"), (imgL_s.size[0], imgL_s.size[1], 3))
            imgR_s = np.reshape(np.array(imgR_s, dtype="uint8"), (imgR_s.size[0], imgR_s.size[1], 3))
            imgYL_s = np.reshape(np.array(imgYL_s, dtype="uint8"), (imgYL_s.size[0], imgYL_s.size[1], 3))
#           imgYR_s = np.reshape(np.array(imgYR_s, dtype="uint8"), (imgYR_s.size[0], imgYR_s.size[1], 3))

            for j in xrange(args.DATA_AUG_TIMES):
                im_h, im_w, _ = imgL_s.shape
                assert im_h == imgR_s.shape[0]
                assert im_w == imgR_s.shape[1]
                assert im_h == imgYL_s.shape[0]
                assert im_w == imgYL_s.shape[1]
#                assert im_h == imgYR_s.shape[0]
#                assert im_w == imgYR_s.shape[1]
                for x in range(0+args.step, im_h-args.pat_size, args.stride):
                    for y in range(0+args.step, im_w-args.pat_size, args.stride):
                        mode = random.randint(0, 7)
                        inputsL[count,:,:,:]  = data_augmentation(imgL_s[x:x+args.pat_size , y:y+args.pat_size, :], mode)
                        inputsR[count,:,:,:]  = data_augmentation(imgR_s[x:x+args.pat_size , y:y+args.pat_size, :], mode)
                        inputsYL[count,:,:,:] = data_augmentation(imgYL_s[x:x+args.pat_size, y:y+args.pat_size, :], mode)
#                        inputsYR[count,:,:,:] = data_augmentation(imgYR_s[x:x+args.pat_size, y:y+args.pat_size, :], mode)
                        count += 1
    if count < numPatches:
        to_pad = numPatches - count
        inputsL[-to_pad:, :,:,:] = inputsL[:to_pad,:,:,:]
        inputsR[-to_pad:, :,:,:] = inputsR[:to_pad,:,:,:]
        inputsYL[-to_pad:, :,:,:] = inputsYL[:to_pad,:,:,:]
#        inputsYR[-to_pad:, :,:,:] = inputsYR[:to_pad,:,:,:]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    np.save(os.path.join(args.save_dir, "pa_L_p{}_b{}_da{}".format(args.pat_size,args.bat_size,args.DATA_AUG_TIMES)), inputsL)
    np.save(os.path.join(args.save_dir, "pa_R_p{}_b{}_da{}".format(args.pat_size,args.bat_size,args.DATA_AUG_TIMES)), inputsR)
    np.save(os.path.join(args.save_dir, "gt_L_p{}_b{}_da{}".format(args.pat_size,args.bat_size,args.DATA_AUG_TIMES)), inputsYL)
#    np.save(os.path.join(args.save_dir, "gt_R_p{}_b{}_da{}".format(args.pat_size,args.bat_size,args.DATA_AUG_TIMES)), inputsYR)

    print("size of inputs tensor = " + str(inputsL.shape))

if __name__ == '__main__':
    generate_patches()
