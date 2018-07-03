#! /usr/bin/env python2

import time
from utils import *
import h5py
from math import ceil
from random import shuffle

def cnn_simple(inL, inR, is_training=True):
    with tf.variable_scope('cnn'):
        out = tf.layers.conv2d(inL, 3, 3, (1,1), activation = tf.nn.relu, padding='same')
        out = tf.layers.conv2d(inL, 3, 3, (1,1), activation = tf.nn.sigmoid, padding='same')
#    for i in range(0,0):
#        out = tf.layers.conv2d(out, 3, 3, (1,1), activation = tf.nn.relu, padding='same')
#    out = tf.layers.conv2d(out, 3, 3, (1,1), padding='same')
    return out

def simple_net(inL, inR, is_training=True):
    out = tf.concat([inL, inR], 3)
    out = tf.layers.conv2d(out, 4, 3, (1, 1), activation=tf.nn.relu, padding='same')
    out = tf.layers.conv2d(out, 3, 3, (1, 1), activation=tf.nn.relu, padding='same')
    return out

def msr_net(inL, inR, n, v, K, is_training=True):
    assert n >= 2, 'n must be greater than 2'
    assert len(v) == n, 'len(v) must be equal to n'
    assert K >= 2, 'K must be greater than 2'
    tf_v = []
    for i in v:
        tf_v.append(tf.constant(i,dtype=tf.float32))
    ### ---------- 1. Multi-scale Log Transformation ---------- ###
    MSL0 = tf.div(tf.log1p(tf.scalar_mul(tf_v[0], inL)), tf.log(1+tf_v[0]))
    MSL1 = tf.div(tf.log1p(tf.scalar_mul(tf_v[1], inL)), tf.log(1+tf_v[1]))
    
    outL = tf.concat([MSL0, MSL1], 3)
    #print("[1L] "+str(outL.shape))
    for i in tf_v[2:]:
        MSLtmp = tf.div(tf.log1p(tf.scalar_mul(i, inL)), tf.log(1+i))
        outL = tf.concat([outL, MSLtmp], 3)
    #print("[2L] "+str(outL.shape))
    outL =  tf.layers.conv2d(outL, 3, 3, (1, 1), activation=tf.nn.relu, padding='same', name='conv1L')
    #print("[3L] "+str(outL.shape))
    outL =  tf.layers.conv2d(outL, 3, 3, (1, 1), padding='same', name='conv2L')
    #print("[4L] "+str(outL.shape))
    ### ---------- 2. Difference of Convolution ---------- ###
    HL = []
    HL.append(outL)
    for k in range(1,K+1):
        HL.append(tf.layers.conv2d(HL[k-1], 32, 3, (1, 1), activation=tf.nn.relu, padding='same', name='convH%dL'%k))
    HL = HL[1:]
    #print("[5L] "+str(len(HL)))
    HK1L = tf.concat([HL[0],HL[1]],3) 
    for k in range(2,K):
        HK1L = tf.concat([HK1L,HL[k]],3)
    #print("[6L] "+str(HK1L.shape))
    HK1L = tf.layers.conv2d(HK1L, 3, 1, (1, 1), padding='same', name='convHK1L')
    #print("[7L] "+str(HK1L.shape))
    outL = tf.subtract(outL, HK1L)
    #print("[8L] "+str(outL.shape))
    ### ---------- 3. Color Restoration Function ---------- ###
    outL = tf.layers.conv2d(outL, 3, 1, (1, 1), activation=tf.nn.sigmoid, padding='same', name='color')
    return outL


def msr_dual_net(inL, inR, n, v, K, is_training=True):
    assert n >= 2, 'n must be greater than 2'
    assert len(v) == n, 'len(v) must be equal to n'
    assert K >= 2, 'K must be greater than 2'
    tf_v = []
    for i in v:
        tf_v.append(tf.constant(i,dtype=tf.float32))
    ### ---------- 1. Multi-scale Log Transformation ---------- ###
    MSL0 = tf.div(tf.log1p(tf.scalar_mul(tf_v[0], inL)), tf.log(1+tf_v[0]))
    MSR0 = tf.div(tf.log1p(tf.scalar_mul(tf_v[0], inR)), tf.log(1+tf_v[0]))
    MSL1 = tf.div(tf.log1p(tf.scalar_mul(tf_v[1], inL)), tf.log(1+tf_v[1]))
    MSR1 = tf.div(tf.log1p(tf.scalar_mul(tf_v[1], inR)), tf.log(1+tf_v[1]))
    
    outL = tf.concat([MSL0, MSL1], 3)
    outR = tf.concat([MSR0, MSR1], 3)
    #print("[1L] "+str(outL.shape))
    for i in tf_v[2:]:
        MSLtmp = tf.div(tf.log1p(tf.scalar_mul(i, inL)), tf.log(1+i))
        MSRtmp = tf.div(tf.log1p(tf.scalar_mul(i, inR)), tf.log(1+i))
        outL = tf.concat([outL, MSLtmp], 3)
        outR = tf.concat([outR, MSRtmp], 3)
    #print("[2L] "+str(outL.shape))
    outL =  tf.layers.conv2d(outL, 3, 3, (1, 1), activation=tf.nn.relu, padding='same', name='conv1L')
    outR =  tf.layers.conv2d(outR, 3, 3, (1, 1), activation=tf.nn.relu, padding='same', name='conv1R')
    #print("[3L] "+str(outL.shape))
    outL =  tf.layers.conv2d(outL, 3, 3, (1, 1), padding='same', name='conv2L')
    outR =  tf.layers.conv2d(outR, 3, 3, (1, 1), padding='same', name='conv2R')
    #print("[4L] "+str(outL.shape))
    ### ---------- 2. Difference of Convolution ---------- ###
    HL = []
    HR = []
    HL.append(outL)
    HR.append(outR)
    for k in range(1,K+1):
        HL.append(tf.layers.conv2d(HL[k-1], 32, 3, (1, 1), activation=tf.nn.relu, padding='same', name='convH%dL'%k))
        HR.append(tf.layers.conv2d(HR[k-1], 32, 3, (1, 1), activation=tf.nn.relu, padding='same', name='convH%dR'%k))
    HL = HL[1:]
    HR = HR[1:]
    #print("[5L] "+str(len(HL)))
    HK1L = tf.concat([HL[0],HL[1]],3) 
    HK1R = tf.concat([HR[0],HR[1]],3) 
    for k in range(2,K):
        HK1L = tf.concat([HK1L,HL[k]],3)
        HK1R = tf.concat([HK1R,HR[k]],3)
    #print("[6L] "+str(HK1L.shape))
    HK1L = tf.layers.conv2d(HK1L, 3, 1, (1, 1), padding='same', name='convHK1L')
    HK1R = tf.layers.conv2d(HK1R, 3, 1, (1, 1), padding='same', name='convHK1R')
    #print("[7L] "+str(HK1L.shape))
    outL = tf.subtract(outL, HK1L)
    outR = tf.subtract(outR, HK1R)
    #print("[8L] "+str(outL.shape))
    ### ---------- 3. Color Restoration Function ---------- ###
    outL = tf.layers.conv2d(outL, 3, 1, (1, 1), padding='same', name='colorL')
    outR = tf.layers.conv2d(outR, 3, 1, (1, 1), padding='same', name='colorR')
    ### ---------- 4. Registration / Concatenation ---------- ###
    out = tf.concat([outL, outR],3)
    #print("[1] "+str(out.shape))
    for i in range(1,4):
        out = tf.layers.conv2d(out, 16, 3, (1, 1), activation=tf.nn.relu, padding='same', name='conv2d%d'%i)
    #print("[2] "+str(out.shape))
    out = tf.layers.conv2d(out, 3, 3, (1, 1), activation=tf.nn.sigmoid, padding='same', name='final')
    #print("[3] "+str(out.shape))
    return out

class imdualenhancer(object):
    def __init__(self, sess, n, v, K, batch_size, model_name, proc, data_path, eval_dir, test_dir, ckpt_dir):
        self.sess = sess
        self.model_name = model_name
        self.proc = proc
        self.batch_size=batch_size
        self.hdf5_file = h5py.File(data_path, "r")
        self.eval_dir = eval_dir
        self.test_dir = test_dir
        self.ckpt_dir = ckpt_dir

        # build model
        with tf.name_scope('input'):
            self.XL = tf.placeholder(tf.float32, [None, None, None, 3], name='left_patch')
            tf.summary.image('inputxl',self.XL, 1)
            self.XR = tf.placeholder(tf.float32, [None, None, None, 3], name='right_patch')
            tf.summary.image('inputxr',self.XR, 1)
            self.Y_ = tf.placeholder(tf.float32, [None, None, None, 3], name='normal_patch')
            tf.summary.image('gtyl',self.Y_, 1)
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        if self.model_name == 'msr_net':
            self.Y = msr_net(self.XL, self.XR, n, v, K, is_training=self.is_training)
        elif self.model_name == 'msr_dual_net':
            self.Y = msr_dual_net(self.XL, self.XR, n, v, K, is_training=self.is_training)
        elif self.model_name == 'simple_net':
            self.Y = simple_net(self.XL, self.XR, is_training=self.is_training)
        elif self.model_name == 'cnn_simple':
            self.Y = cnn_simple(self.XL, self.XR, is_training=self.is_training)
        else:
            print('Model name is not correct.')
            sys.exit(1)
        self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y - self.Y_)
        #self.loss = (1.0 / self.batch_size) * tf.losses.absolute_difference(self.Y_, self.Y, weights=1.0)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully ...")
        sys.stdout.flush()

    def train(self, end_epoch, lr, eval_every_epoch=2):
        data_num = self.hdf5_file["XL_tr"].shape[0]
        numBatch = int(data_num / self.batch_size)
        print("[*] Training on %d/%d samples/batches." % (data_num, numBatch))
        ## load pretrained model
        #load_model_status, global_step = self.load(self.ckpt_dir)
        #if load_model_status:
        #    iter_num = global_step
        #    start_epoch = global_step // numBatch
        #    start_step = global_step % numBatch
        #    print("[*] Model restore success!")
        #    sys.stdout.flush()
        #else:
        iter_num = 0
        start_epoch = 0
        #start_step = 0
        #print("[*] Not find pretrained model!")
        #sys.stdout.flush()

        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        logdir = './logs/'+self.model_name+'-'+self.proc+'/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = tf.summary.FileWriter(logdir, self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d: " % start_epoch)
        sys.stdout.flush()
        start_time = time.time()
        self.evaluate(iter_num, summary_merged=summary_psnr, summary_writer=writer)
        batches_list = list(range(int(ceil(float(data_num)/self.batch_size))))
        for epoch in xrange(start_epoch, end_epoch):
            shuffle(batches_list)

            loss = np.zeros(len(batches_list))
            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size
                i_e = min([(i+1)*self.batch_size, data_num])
                batch_XL = self.hdf5_file["XL_tr"][i_s:i_e, ...].astype(np.float32) / 255.0
                batch_XR = self.hdf5_file["XR_tr"][i_s:i_e, ...].astype(np.float32) / 255.0
                batch_YL = self.hdf5_file["YL_tr"][i_s:i_e, ...].astype(np.float32) / 255.0

                _, loss[i], summary = self.sess.run(
                        [self.train_op, self.loss, merged],
                        feed_dict={self.Y_:batch_YL,
                                   self.XL:batch_XL,
                                   self.XR:batch_XR,
                                   self.lr: lr[epoch], ###!!!
                                   self.is_training:True})
                iter_num += 1
                writer.add_summary(summary, iter_num)
            print("Epoch: [%2d] batch: [%3d] time: %4.4f, avg_loss: %.6f"
                     % (epoch+1, len(batches_list), time.time() - start_time, loss.mean()))
            sys.stdout.flush()
            if np.mod(epoch+1, eval_every_epoch) == 0:
                self.evaluate(iter_num, summary_merged=summary_psnr, summary_writer=writer)
                self.save(iter_num)
        print("[*] Finish training.")
        sys.stdout.flush()
        self.hdf5_file.close()
        writer.close()

    def evaluate(self, iter_num, summary_merged, summary_writer):
        print("[*] Evaluating...")
        sys.stdout.flush()

        psnr_sum = 0
        eval_num = self.hdf5_file["XL_eval"].shape[0]
        assert self.hdf5_file["XL_eval"].shape[3] == 3

        for idx in xrange(eval_num):
            imL_tmp = self.hdf5_file["XL_eval"][idx, ...].astype(np.float32)
            imR_tmp = self.hdf5_file["XR_eval"][idx, ...].astype(np.float32)
            gt_tmp  = self.hdf5_file["YL_eval"][idx, ...].astype(np.float32)
            imL = imL_tmp.reshape(1, imL_tmp.shape[0], imL_tmp.shape[1], imL_tmp.shape[2]) / 255.0
            imR = imR_tmp.reshape(1, imR_tmp.shape[0], imR_tmp.shape[1], imR_tmp.shape[2]) / 255.0
            gt  = gt_tmp.reshape( 1, gt_tmp.shape[0] , gt_tmp.shape[1] , gt_tmp.shape[2] ) / 255.0
            imOut, psnr_summary = self.sess.run([self.Y, summary_merged],
                                  feed_dict={self.Y_:gt, self.XL:imL,
                                             self.XR:imR, self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(gt_tmp   , 0, 255).astype('uint8')
            dark_imageL = np.clip(imL_tmp  , 0, 255).astype('uint8')
            output_image= np.clip(255*imOut, 0, 255).astype('uint8')

            # calculate PSNR
            psnr = cal_psnr(groundtruth, output_image)
            print("img%d PSNR: %.2f" % (idx+1, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            if idx == 0:
                save_images(os.path.join(self.eval_dir, 'eval_%s_%s_%d_%d.png' % (self.model_name, self.proc, idx+1, iter_num)),
                        groundtruth, dark_imageL, output_image)
        avg_psnr = psnr_sum / eval_num
        print("--- Evaluation --- Average PSNR %.3f ---" % avg_psnr)
        sys.stdout.flush()


    def save(self, iter_num):
        saver = tf.train.Saver()
        checkpoint_dir = self.ckpt_dir
        print("[*] Saving model...")
        sys.stdout.flush()
        saver.save(self.sess,
                   os.path.join(checkpoint_dir,
                   self.model_name+'-'+self.proc),
                   global_step=iter_num)

    #def load(self, checkpoint_dir):
    #    print("[*] Reading checkpoint...")
    #    sys.stdout.flush()
    #    saver = tf.train.Saver()
    #    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #    if ckpt and ckpt.model_checkpoint_path:
    #        full_path = tf.train.latest_checkpoint(checkpoint_dir)
    #        global_step = int(full_path.split('/')[-1].split('-')[-1])
    #        saver.restore(self.sess, full_path)
    #        return True, global_step
    #    else:
    #        return False, 0

    #def test(self, test_filesL, test_filesR, test_filesY,
    #         ckpt_dir, save_dir):
    #    # init variables
    #    tf.initialize_all_variables().run()
    #    assert len(test_filesL) != 0, 'No testing data!'
    #    load_model_status, global_step = self.load(ckpt_dir)
    #    assert load_model_status == True, '[!] Load weights FAILED...'
    #    print("[*] Load weights SUCCESS...")
    #    sys.stdout.flush()
    #    psnr_sum = 0
    #    print("[*] Start testing...")
    #    sys.stdout.flush()
    #    for idx in xrange(len(test_filesL)):
    #        imL = load_images(test_filesL[idx],0.5,0.5).astype(np.float32) / 255.0
    #        imR = load_images(test_filesR[idx],0.5,0.5).astype(np.float32) / 255.0
    #        imY = load_images(test_filesY[idx],0.5,0.5).astype(np.float32) / 255.0
    #        Yhat = self.sess.run(
    #                [self.Y], feed_dict={
    #                    self.Y_:imY,
    #                    self.XL:imL,
    #                    self.XR:imR,
    #                    self.is_training: Flase})
    #        groundtruth = np.clip(255*imY , 0, 255).astype('uint8')
    #        inputL      = np.clip(255*imL , 0, 255).astype('uint8')
    #        outputimage = np.clip(255*Yhat, 0, 255).astype('uint8')
    #        # calculate PSNR
    #        psnr = cal_psnr(groundtruth, outputimage)
    #        print("img%d PSNR: %.2f" % (idx, psnr))
    #        sys.stdout.flush()
    #        psnr_sum += psnr
    #        save_images(os.path.join(save_dir, 'inputL%d.png' % idx),
    #                inputL)
    #        save_images(os.path.join(save_dir, 'output%d.png' % idx),
    #                outputimage)
    #    avg_psnr = psnr_sum / len(test_filesL)
    #    print("--- Average PSNR %.2f ---" % avg_psnr)
    #    sys.stdout.flush()
