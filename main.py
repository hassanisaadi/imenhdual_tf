#! /usr/bin/env python2


import argparse
from glob import glob
import tensorflow as tf
from model import imdualenhancer
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=0, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample_results', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test_results', help='test sample are saved here')
parser.add_argument('--n', dest='n', type=int, default=2, help='number of multi-scale filters')
parser.add_argument('--v', dest='v', default='1,10', help='')
parser.add_argument('--K', dest='K', type=int, default=2, help='number of filters')
parser.add_argument('--fpxl', dest='fpxl', default='./data/mb2014_bin/pa_L_p8_b16_da1.npy', help='filepath to x_left')
parser.add_argument('--fpxr', dest='fpxr', default='./data/mb2014_bin/pa_R_p8_b16_da1.npy', help='filepath to x_right')
parser.add_argument('--fpyl', dest='fpyl', default='./data/mb2014_bin/gt_L_p8_b16_da1.npy', help='filepath to y_left')
#parser.add_argument('--fpyr', dest='fpyr', default='./data/mb2014_bin/gt_R_p8_b16_da1.np', help='filepath to y_right')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=10, help='evaluate every epoch')



args = parser.parse_args()

args.v = [np.float32(s) for s in args.v.split(',')]

def imdualenhancer_train(imdualenhancer, lr, proc):
    with load_data(fxl=args.fpxl,
                   fxr=args.fpxr,
                   fyl=args.fpyl) as data:
        # if there is a small memory, please comment this line and uncomment the line99 in model.py
        XL = data[0].astype(np.float32) / 255.0  # normalize the data to 0-1
        XR = data[1].astype(np.float32) / 255.0 
        YL = data[2].astype(np.float32) / 255.0  
        eval_filesXL = glob('./data/mb2014_png/eval/X_left/*.png')
        eval_filesXR = glob('./data/mb2014_png/eval/X_right/*.png')
        eval_filesYL = glob('./data/mb2014_png/eval/Y_left/*.png')
        eval_dataXL = load_images(eval_filesXL,0.3,0.3)[0:1]  # list of array of different size, 4-D, pixel value range is 0-255
        eval_dataXR = load_images(eval_filesXR,0.3,0.3)[0:1]
        eval_dataYL = load_images(eval_filesYL,0.3,0.3)[0:1]
        imdualenhancer.train(XL, XR, YL, 
                eval_dataXL, eval_dataXR, eval_dataYL,
                batch_size=args.batch_size, 
                ckpt_dir=args.ckpt_dir, end_epoch=args.epoch, lr=lr,
                sample_dir=args.sample_dir,
                eval_every_epoch=args.eval_every_epoch, proc=proc)


def imdualenhancer_test(denoiser):
    test_filesXL = glob('./data/mb2014_png/test/X_left/*.png')
    test_filesXR = glob('./data/mb2014_png/test/X_right/*.png')
    test_filesYL = glob('./data/mb2014_png/test/Y_left/*.png')
    imdualenhancer.test(test_filesXL, test_filesXR, test_filesYL,
            ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[1000:] = lr[0] / 10.0
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        sys.stdout.flush()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = imdualenhancer(sess, args.n, args.v, args.K, args.batch_size)
            if args.phase == 'train':
                imdualenhancer_train(model, lr=lr, proc='gpu')
            elif args.phase == 'test':
                imdualenhancer_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        sys.stdout.flush()
        with tf.Session() as sess:
            model = imdualenhancer(sess, args.n, args.v, args.K, args.batch_size)
            if args.phase == 'train':
                imdualenhancer_train(model, lr=lr, proc='cpu')
            elif args.phase == 'test':
                imdualenhancer_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
