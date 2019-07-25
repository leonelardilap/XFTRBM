from rbm_class.rbm import RBM 
from rbm_class.rbm import save_images 
from rbm_class.dataset import DataSet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import random
import gzip, struct

logs_dir = './logs'

rbm = RBM()

import csv
#SAVE THE DATA
def Save_Data(data, name):
    ''' Save a list data in a .csv file '''
    out = csv.writer(open(name,'a'),delimiter=',', quoting=csv.QUOTE_ALL)
    for i in data:
        out.writerow([i])

#RESTORE SAVE DATA
def Restore_data(file_name):
    ''' Restore the information .csv file in a python list '''
    data=[]
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(float(list(row)[0]))
        #reader = csv.reader(f)
        #restored_list = list(reader)[0]
        #data = [float(i) for i in restored_list]
        return data


def Test_XFT_Theorem(rbm,temperature2,amount_samples,gibbs_steps,model_name,amount_data):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, rbm.n_visible])
        y = tf.placeholder(tf.float32, shape=[None, rbm.n_hidden])
        saver = tf.train.Saver()
        saver.restore(sess,model_name)
        gibbs_1 = rbm.Gibbs_Sampling(x,gibbs_steps)
        gibbs_2 = rbm.Gibbs_Sampling(x,1)
        energy = rbm.energy(x,y)
        energy_variations = []
        beta2 = 1.0/temperature2
        for i in range(amount_samples//amount_data):
            initial_state = np.random.rand(amount_data,rbm.n_visible)
            #Thermalization
            visible_1,hidden_1 = sess.run(gibbs_1,feed_dict={x:np.floor(initial_state+0.5),rbm.beta:1.0})
            energy_1 = sess.run(energy,feed_dict = {x:visible_1,y:hidden_1})
            #Thermal contac with bath at temperature2
            visible_2,hidden_2 = sess.run(gibbs_2,feed_dict={x:visible_1,rbm.beta:beta2})
            energy_2 = sess.run(energy,feed_dict = {x:visible_2,y:hidden_2})
            delta_energy = energy_2 - energy_1
            energy_substep = []
            for m in delta_energy:
                energy_variations.append(float(m))
                energy_substep.append(float(m))
                #print(m)
            name_file = './Data/Random_Weights_Delta_Energy_Temperature_'+str(temperature2)+'_sub_step_'+str(i)+'.csv'
            Save_Data(energy_substep,name_file)

    name_file_2 = './Data/Random_Weights_DE_T_'+str(temperature2)+'.csv'
    Save_Data(energy_variations,name_file_2)        
    return energy_variations


Amount_Samples = 2000000
Gibbs_Steps = 300
Amount_Data = 100
Model_Name = './logs/RBM_Random_Weights-2'


Temperature = 1.1
energy_variations_1 = Test_XFT_Theorem(rbm,Temperature,Amount_Samples,Gibbs_Steps,Model_Name,Amount_Data)

Temperature = 1.2
energy_variations_2 = Test_XFT_Theorem(rbm,Temperature,Amount_Samples,Gibbs_Steps,Model_Name,Amount_Data)

Temperature = 0.9
energy_variations_3 = Test_XFT_Theorem(rbm,Temperature,Amount_Samples,Gibbs_Steps,Model_Name,Amount_Data)    

Temperature = 0.8
energy_variations_4 = Test_XFT_Theorem(rbm,Temperature,Amount_Samples,Gibbs_Steps,Model_Name,Amount_Data)
