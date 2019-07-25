import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

class RBM:
    i = 0 # fliping index for computing pseudo likelihood
    
    def __init__(self, n_visible=784, n_hidden=500, momentum=False):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.beta = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        if momentum:
            self.momentum = tf.placeholder(tf.float32)
        else:
            self.momentum = 0.0
        
        self.w = tf.Variable(tf.truncated_normal([n_visible, n_hidden],mean=0.0, stddev=0.5,seed=10), name='w') 
        self.hb = tf.Variable(tf.truncated_normal(shape=[n_hidden],mean=0.0,stddev=0.5,seed=2), name='hb') 
        self.vb = tf.Variable(tf.truncated_normal(shape=[n_visible],mean=0.0,stddev=0.5,seed=4), name='vb') 
        
        self.w_v = tf.Variable(tf.zeros([n_visible, n_hidden]), dtype=tf.float32)
        self.hb_v = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)
        self.vb_v = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
        
    def propup(self, visible):
        pre_sigmoid_activation = self.beta*tf.matmul(visible, self.w) + self.beta*self.hb
        return tf.nn.sigmoid(pre_sigmoid_activation)
    
    def propdown(self, hidden):
        pre_sigmoid_activation = self.beta*tf.matmul(hidden, tf.transpose(self.w)) + self.beta*self.vb
        return tf.nn.sigmoid(pre_sigmoid_activation)
    
    def sample_h_given_v(self, v_sample):
        h_props = self.propup(v_sample)
        h_sample = tf.nn.relu(tf.sign(h_props - tf.random_uniform(tf.shape(h_props))))
        return h_sample
    
    def sample_v_given_h(self, h_sample):
        v_props = self.propdown(h_sample)
        v_sample = tf.nn.relu(tf.sign(v_props - tf.random_uniform(tf.shape(v_props))))
        return v_sample
    
    def Gibbs_Sampling(self,visibles,steps):
        # k steps gibbs sampling
        v_samples = visibles
        h_samples = self.sample_h_given_v(v_samples)
        for i in range(steps):
            v_samples = self.sample_v_given_h(h_samples)
            h_samples = self.sample_h_given_v(v_samples)
        return v_samples, h_samples
    
    def CD_k(self, visibles,steps):       
        # k steps gibbs sampling
        #v_samples = visibles
        #h_samples = self.sample_h_given_v(v_samples)
        #for i in range(self.k):
        #    v_samples = self.sample_v_given_h(h_samples)
        #    h_samples = self.sample_h_given_v(v_samples)
        
        v_samples, h_samples = self.Gibbs_Sampling(visibles,steps)
        
        h0_props = self.propup(visibles)
        w_positive_grad = tf.matmul(tf.transpose(visibles), h0_props)
        w_negative_grad = tf.matmul(tf.transpose(v_samples), h_samples)
        w_grad = self.beta*(w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(visibles)[0])
        hb_grad = self.beta*tf.reduce_mean(h0_props - h_samples, 0)
        vb_grad = self.beta*tf.reduce_mean(visibles - v_samples, 0)
        return w_grad, hb_grad, vb_grad
    
    def learn(self, visibles,steps):
        w_grad, hb_grad, vb_grad = self.CD_k(visibles,steps)
        # compute new velocities
        new_w_v = self.momentum * self.w_v + self.lr * w_grad
        new_hb_v = self.momentum * self.hb_v + self.lr * hb_grad
        new_vb_v = self.momentum * self.vb_v + self.lr * vb_grad
        # update parameters
        update_w = tf.assign(self.w, self.w + new_w_v)
        update_hb = tf.assign(self.hb, self.hb + new_hb_v)
        update_vb = tf.assign(self.vb, self.vb + new_vb_v)
        # update velocities
        update_w_v = tf.assign(self.w_v, new_w_v)
        update_hb_v = tf.assign(self.hb_v, new_hb_v)
        update_vb_v = tf.assign(self.vb_v, new_vb_v)
        
        return [update_w, update_hb, update_vb, update_w_v, update_hb_v, update_vb_v]
        
    def sampler(self, visibles, steps=5000):
        v_samples = visibles
        for step in range(steps):
            v_samples = self.sample_v_given_h(self.sample_h_given_v(v_samples))
        return v_samples
    
    def free_energy(self, visibles):
        first_term = tf.matmul(visibles, tf.reshape(self.vb, [tf.shape(self.vb)[0], 1]))
        second_term = tf.reduce_sum(tf.log(1 + tf.exp(self.beta*(self.hb + tf.matmul(visibles, self.w)))), axis=1)
        return - first_term - (second_term/self.beta)
    
    def energy(self,visibles,hidden):
        first_term = tf.matmul(visibles, tf.reshape(self.vb, [tf.shape(self.vb)[0], 1]))
        second_term = tf.matmul(hidden, tf.reshape(self.hb, [tf.shape(self.hb)[0], 1]))
        term_3 = tf.diag_part(tf.matmul(visibles, tf.matmul(self.w, tf.transpose(hidden))))
        third_term = tf.reshape(term_3,[tf.shape(term_3)[0],1])
        return - first_term - second_term - third_term
        #return -third_term
    
    def pseudo_likelihood(self, visibles):
        x = tf.round(visibles)
        x_fe = self.free_energy(x)
        split0, split1, split2 = tf.split(x, [self.i, 1, tf.shape(x)[1] - self.i - 1], 1)
        xi = tf.concat([split0, 1 - split1, split2], 1)
        self.i = (self.i + 1) % self.n_visible
        xi_fe = self.free_energy(xi)
        return tf.reduce_mean(self.n_visible * tf.log(tf.nn.sigmoid(xi_fe - x_fe)), axis=0)


#Save image
import scipy.misc

# 保存图片
def save_images(images, size, path):
    
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """
    
    # 图片归一化，主要用于生成器输出是tanh形式的归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    
    # 生成一个大画布，用来保存生成的batch_size个图像
    merge_img = np.zeros((h * size[0], w * size[1]))
    
    # 循环把画布各个位置的值赋为batch里各幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w] = image
        
    # 保存画布
    return scipy.misc.imsave(path, merge_img)



#def weight(shape, name='weights'):
#    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

#def bias(shape, name='biases'):
#    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


"""
logs_dir = './logs'
samples_dir = './samples'
    
x = tf.placeholder(tf.float32, shape=[None, 784])
noise_x, _ = test_data.sample_batch()
noise_x = tf.random_normal([train_data.batch_size, 784])
rbm = RBM()
step = rbm.learn(x)
sampler = rbm.sampler(x)
pl = rbm.pseudo_likelihood(x)
    
saver = tf.train.Saver()
    
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    mean_cost = []
    epoch = 1
    for i in range(epoches): #* train_data.batch_num):
        # draw samples
        if i % 20 == 0:
            samples = sess.run(sampler, feed_dict = {x: np.floor(noise_x+0.5)})
            samples = samples.reshape([train_data.batch_size, 28, 28])
            save_images(samples, [8, 8], os.path.join(samples_dir, 'iteration_%d.png' % i))
            print('Saved samples.')
        print(i)
        batch_x, _ = train_data.next_batch()
        sess.run(step, feed_dict = {x: np.floor(batch_x+0.5), rbm.lr: 0.1})
        epoch +=1
        cost = sess.run(pl, feed_dict = {x: np.floor(batch_x+0.5)})
        mean_cost.append(cost)
        # save model
        if i is not 0 and train_data.batch_index is 0:
            checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step = epoch + 1)
            print('Saved Model.')
         print pseudo likelihood
        if i is not 0 and train_data.batch_index is 0:
            print('Epoch %d Cost %g' % (epoch, np.mean(mean_cost)))
            mean_cost = []
            epoch += 1
    print('Test')
    samples = sess.run(sampler, feed_dict = {x: np.floor(noise_x+0.5)})
    samples = samples.reshape([train_data.batch_size, 28, 28])
    save_images(samples, [8, 8], os.path.join(samples_dir, 'test_Juan.png'))
    print('Saved samples.')
"""