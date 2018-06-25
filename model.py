#! /usr/bin/env python2

import time
from utils import *

def msr_dual_net(inL, inR, n, v, K, is_training=True):
    ### Left
    assert n >= 2, 'n must be greater than 2'
    assert len(v) == n, 'len(v) must be equal to n'
    tf_v = []
    for i in v:
        tf_v.append(tf.constant(i,dtype=tf.float32))
    MSL0 = tf.div(tf.log1p(tf.scalar_mul(tf_v[0], inL)), tf.log(1+tf_v[0]))
    MSL1 = tf.div(tf.log1p(tf.scalar_mul(tf_v[1], inL)), tf.log(1+tf_v[1]))

    outL = tf.concat([MSL0, MSL1], 3) 
    for i in tf_v[2:]:
        MSLtmp = tf.div(tf.log1p(tf.scalar_mul(i, inL)),
                      tf.log(1+i))
        outL = tf.concat([outL, MSLtmp], 3)

    outL =  tf.layers.conv2d(
            outL, 32 , 1, (1, 1), activation=tf.nn.relu,
            padding='same', name='conv1L')
    outL =  tf.layers.conv2d(
            outL, 3 , 3, (1, 1), activation=tf.nn.relu,
            padding='same', name='conv2L')
    HL = []
    HL.append(tf.layers.conv2d(
            outL, 32, 3, (1, 1), activation=tf.nn.relu,
            padding='same', name='convH0L'))
    for k in range(1,K):
        HL.append(tf.layers.conv2d(
                HL[k-1], 32, 3, (1, 1), activation=tf.nn.relu,
                padding='same', name='convH%dL'%k))
    HK1L = tf.concat([HL[0],HL[1]],3) 
    for k in range(2,K):
        HK1L = tf.concat([HK1L,HL[k]],3)
    HK1L = tf.layers.conv2d(
        HK1L, 3, 1, (1, 1), activation=tf.nn.relu,
        padding='same', name='convHK1L')
    outL = tf.subtract(outL, HK1L)
    ### Right
    MSR0 = tf.div(tf.log1p(tf.scalar_mul(tf_v[0], inR)),
                  tf.log(1+tf_v[0]))
    MSR1 = tf.div(tf.log1p(tf.scalar_mul(tf_v[1], inR)),
                  tf.log(1+tf_v[1]))

    outR = tf.concat([MSR0, MSR1], 3)
    for i in tf_v[2:]:
        MSRtmp = tf.div(tf.log1p(tf.scalar_mul(i, inR)),
                      tf.log(1+i))
        outR = tf.concat([outR, MSRtmp], 3)

    outR =  tf.layers.conv2d(
            outR, 32 , 1, (1, 1), activation=tf.nn.relu,
            padding='same', name='conv1R')
    outR =  tf.layers.conv2d(
            outR, 3 , 3, (1, 1), activation=tf.nn.relu,
            padding='same', name='conv2R')
    HR = []
    HR.append(tf.layers.conv2d(
            outR, 32, 3, (1, 1), activation=tf.nn.relu,
            padding='same', name='convH0R'))
    for k in range(1,K):
        HR.append(tf.layers.conv2d(
                HR[k-1], 32, 3, (1, 1), activation=tf.nn.relu,
                padding='same', name='convH%dR'%k))
    HK1R = tf.concat([HR[0],HR[1]],3)
    for k in range(2,K):
        HK1R = tf.concat([HK1R,HR[k]],3)
    HK1R = tf.layers.conv2d(
        HK1R, 3, 1, (1, 1), activation=tf.nn.relu,
        padding='same', name='convHK1R')
    outR = tf.subtract(outR, HK1R)
    ### Concatenation
    out = tf.concat([outL, outR],3)
    for i in range(1,4):
        out = tf.layers.conv2d(
            out, 16, 3, (1, 1), activation=tf.nn.relu,
            padding='same', name='conv2d%d'%i)
    out = tf.layers.conv2d(
        out, 3, 3, (1, 1), padding='same', name='final')
    return out

class imdualenhancer(object):
    def __init__(self, sess, n, v, K, batch_size=128):
        self.sess = sess
        
        # build model
        self.Y_ = tf.placeholder(tf.float32,
                [None, None, None, 3], name='normal_patch')
        self.XL = tf.placeholder(tf.float32,
                [None, None, None, 3], name='left_patch')
        self.XR = tf.placeholder(tf.float32,
                [None, None, None, 3], name='right_patch')
        self.is_training = tf.placeholder(tf.bool, 
                           name='is_training')
        self.Y = msr_dual_net(self.XL, self.XR, n, v, K, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y - self.Y_)
        self.lr = tf.placeholder(tf.float32, 
                  name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr,
                    name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully ...")
        sys.stdout.flush()

    def evaluate(self, iter_num, testXL, testXR, testY,
                 sample_dir, summary_merged, summary_writer):
        # assert test_Data value range is 0-255
        print("[*] Evaluating...")
        sys.stdout.flush()

        psnr_sum = 0
        for idx in xrange(len(testXL)):
            imL = testXL[idx].reshape(1,testXL[idx].shape[1],testXL[idx].shape[2],testXL[idx].shape[3]).astype(np.float32) / 255.0
            imR = testXR[idx].reshape(1,testXR[idx].shape[1],testXR[idx].shape[2],testXR[idx].shape[3]).astype(np.float32) / 255.0
            gt  = testY[idx].reshape(1 ,testY[idx].shape[1],testY[idx].shape[2],testY[idx].shape[3]).astype(np.float32) / 255.0
            imOut, psnr_summary = self.sess.run(
                    [self.Y, summary_merged],
                    feed_dict={self.Y_:gt, self.XL:imL,
                               self.XR:imR, self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(testY[idx] , 0, 255).astype('uint8')
            dark_imageL = np.clip(testXL[idx], 0, 255).astype('uint8')
            output_image= np.clip(255*imOut  , 0, 255).astype('uint8')

            # calculate PSNR
            psnr = cal_psnr(groundtruth, output_image)
            print("img%d PSNR: %.2f" % (idx+1, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx+1, iter_num)),
                        groundtruth, dark_imageL, output_image)
        avg_psnr = psnr_sum / len(testXL)
        print("--- Test --- Average PSNR %.2f ---" % avg_psnr)
        sys.stdout.flush()


    def train(self, XL, XR, Y, 
              eval_XL, eval_XR, eval_Y,
              batch_size, ckpt_dir, end_epoch, lr, 
              sample_dir, eval_every_epoch=2, proc='cpu'):
        # assert data range is between 0 and 1
        numBatch = int(XL.shape[0] / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
            sys.stdout.flush()
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
            sys.stdout.flush()

        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " %
                (start_epoch, iter_num))
        sys.stdout.flush()
        start_time = time.time()
        self.evaluate(iter_num, eval_XL, eval_XR, eval_Y, 
                      sample_dir=sample_dir,
                      summary_merged=summary_psnr,
                      summary_writer=writer) # eval_dat range is 0-255
        for epoch in xrange(start_epoch, end_epoch):
            idx = np.random.permutation(XL.shape[0])
            XL = XL[idx].reshape(XL.shape)
            XR = XR[idx].reshape(XR.shape)
            loss = np.zeros(numBatch)
            for batch_id in xrange(start_step, numBatch):
                batch_imL = XL[batch_id*batch_size:(batch_id+1)*batch_size,:,:,:]
                batch_imR = XR[batch_id*batch_size:(batch_id+1)*batch_size,:,:,:]
                batch_gt  =  Y[batch_id*batch_size:(batch_id+1)*batch_size,:,:,:]
                _, loss[batch_id], summary = self.sess.run(
                        [self.train_op, self.loss, merged],
                        feed_dict={self.Y_:batch_gt, 
                                   self.XL:batch_imL,
                                   self.XR:batch_imR,
                                   self.lr: lr[epoch],
                                   self.is_training: True})
                iter_num += 1
                writer.add_summary(summary, iter_num)
            print("Epoch: [%2d] time: %4.4f, avg_loss: %.6f, std_loss: %.6f"
                  % (epoch+1, time.time() - start_time, loss.sum()/(numBatch-start_step), loss.std())) ###!!! std
            sys.stdout.flush()
            if np.mod(epoch+1, eval_every_epoch) == 0:
               self.evaluate(iter_num, eval_XL, eval_XR, eval_Y,
                       sample_dir=sample_dir,
                       summary_merged=summary_psnr,
                       summary_writer=writer) # eval_data value range is 0-255
               self.save(iter_num, ckpt_dir, proc=proc)
        print("[*] Finish training.")
        sys.stdout.flush()

    def save(self, iter_num, ckpt_dir, model_name='msr_dual_net', proc='cpu'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        sys.stdout.flush()
        saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name+'-'+proc),
                global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        sys.stdout.flush()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_filesL, test_filesR, test_filesY,
             ckpt_dir, save_dir):
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_filesL) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        sys.stdout.flush()
        psnr_sum = 0
        print("[*] Start testing...")
        sys.stdout.flush()
        for idx in xrange(len(test_filesL)):
            imL = load_images(test_filesL[idx],0.5,0.5).astype(np.float32) / 255.0
            imR = load_images(test_filesR[idx],0.5,0.5).astype(np.float32) / 255.0
            imY = load_images(test_filesY[idx],0.5,0.5).astype(np.float32) / 255.0
            Yhat = self.sess.run(
                    [self.Y], feed_dict={
                        self.Y_:imY,
                        self.XL:imL,
                        self.XR:imR,
                        self.is_training: Flase})
            groundtruth = np.clip(255*imY , 0, 255).astype('uint8')
            inputL      = np.clip(255*imL , 0, 255).astype('uint8')
            outputimage = np.clip(255*Yhat, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'inputL%d.png' % idx),
                    inputL)
            save_images(os.path.join(save_dir, 'output%d.png' % idx),
                    outputimage)
        avg_psnr = psnr_sum / len(test_filesL)
        print("--- Average PSNR %.2f ---" % avg_psnr)
        sys.stdout.flush()
