from rbm_class.rbm import RBM 
from rbm_class.rbm import save_images 
from rbm_class.dataset import DataSet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import random
import gzip, struct


train_dir = {
    'X': './mnist/train-images-idx3-ubyte.gz', 
    'Y': './mnist/train-labels-idx1-ubyte.gz'
}

test_dir = {
    'X': './mnist/t10k-images-idx3-ubyte.gz', 
    'Y': './mnist/t10k-labels-idx1-ubyte.gz'
}

logs_dir = './logs'
samples_dir = './samples'

rbm = RBM()

train_data = DataSet(data_dir=train_dir, batch_size=128, one_hot=True)
test_data = DataSet(data_dir=test_dir, batch_size=128, one_hot=True)

def train(train_data,epoches,rbm,model_name,temperatue,gibbs_steps):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    step = rbm.learn(x,gibbs_steps)
    pl = rbm.pseudo_likelihood(x)
    saver = tf.train.Saver()
    beta = 1.0/temperatue
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        mean_cost = []
        epoch = 1
        for ii in range(epoches): 
            batch_x, _ = train_data.next_batch()
            sess.run(step, feed_dict = {x: np.floor(batch_x+0.5), rbm.lr: 0.1, rbm.beta: beta})
            cost = sess.run(pl, feed_dict = {x: batch_x, rbm.beta: beta})
            mean_cost.append(cost)
            # save model
            if ii%2000 == 0:
                checkpoint_path = os.path.join(logs_dir, model_name)
                saver.save(sess, checkpoint_path, global_step = epoch + 1)
                print('Epoch %d'%(ii))
                print('Saved Model.')
                print('Cost %g' % (np.mean(mean_cost)))
            if ii %200 == 0:
                print('Epoch %d Cost %g' % (ii, np.mean(mean_cost)))
                #mean_cost = []
                #epoch += 1
            epoch +=1
        checkpoint_path = os.path.join(logs_dir, model_name)
        saver.save(sess, checkpoint_path, global_step = epoch + 1)
        #print('Saved Model.')
        #print('Cost %g' % (np.mean(mean_cost)))

Model_Name = 'RBM_Random_Weights'
train(train_data,0,rbm,Model_Name,1.0,30)
