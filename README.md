# imenhdual_tf

# Image Enhacement in Dual-Camera Architecture Using Deep Learning (Tensorflow)

## Summary
The goal in this project is enhancing image quality in low-light condition using dual camera architecture. We use a CNN network to train our model on [middleburry](http://vision.middlebury.edu/stereo/data/scenes2014/) dataset. This dataset is for depth estimation using two cameras. Since the dataset provided us image paris (left and right) with different exposure time, we have very dark (low-light) image pairs.

## Prerequisites
You need tensorflow 1.4.1 both for gpu and cpu. For newer version, it throws segmentation fault which I'll fix this in the near future. To install a previous version of tensorflow use pip:
```
pip install tensorflow-cpu==1.4.1
```
or
```
pip install tensorflow-gpu==1.4.1
```

You also need numpy and Image from pillow library.

## Results
This repository is under progress and the results will be updated...

## Inspiration code
Thanks to [DnCNN-tensorflow](https://github.com/crisb-DUT/DnCNN-tensorflow).
