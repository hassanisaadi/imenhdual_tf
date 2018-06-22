#! /usr/bin/env python2

import numpy as np
import os
import subprocess

odirxleft  = './data/mb2014/X_left'
odirxright = './data/mb2014/X_right'
odiryleft  = './data/mb2014/Y_left'
odiryright = './data/mb2014/Y_right'

### 2014 dataset ###
im_counter = 1
base1 = '../scenes2014'
for dir in sorted(os.listdir(base1)):
    print(dir.split('0')[0])

    base2_imperfect = os.path.join(base1, dir)

    
    base3 = os.path.join(base2_imperfect, 'ambient')
    num_light = len(os.listdir(base3))

    for l in range(num_light):
        imgs = []
        for fname in sorted(os.listdir(base3 + '/L{}/dark'.format(l+1))):
            base4 = os.path.join(base3, 'L{}/dark'.format(l+1))
            cam = int(fname[2])
            if cam == 0:
                cam = 1
                exp = fname[4]
                fname2 = fname[0:2] + str(cam) + 'e' + exp + fname[5:]
                if not os.path.isfile(base4 + '/' + fname2):
                    continue
 
                subprocess.check_call('cp {} {}/Y{}.png'.format(os.path.join(base2_imperfect, 'im0.png'), odiryleft , im_counter), shell=True)
                subprocess.check_call('cp {} {}/Y{}.png'.format(os.path.join(base2_imperfect, 'im1.png'), odiryright, im_counter), shell=True)

                subprocess.check_call('cp {} {}/X{}.png'.format(os.path.join(base4, fname) , odirxleft , im_counter), shell=True)
                subprocess.check_call('cp {} {}/X{}.png'.format(os.path.join(base4, fname2), odirxright, im_counter), shell=True)
                im_counter += 1
print('MB 2014 dataset created!')

