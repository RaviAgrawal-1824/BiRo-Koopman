# 9-6-64-3 reduced input state dimension to 21, removed scaling in loss function, lr increased decaying, adam optim, Koop_arch_9-6-64-3.pth
# changed batch size, decrease epochs, increase model with L1 norm, 
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import itertools
import torch
import random
import time
import sys,os
import pickle

# import Koopman Libraries
from core.koopman_core_linear import KoopDNN_linear, KoopmanNet_linear, KoopmanNetCtrl_linear
from core.koopman_core import KoopDNN, KoopmanNet, KoopmanNetCtrl
from sklearn import preprocessing
from core.util import fit_standardizer
from models.koop_model import model_matricies,lift

# Import Adaptive Koopman
from core.adapt_net_linear import AdaptNet_linear
import scipy.io

# num_steps = 500
# num_traj = 500
# num_states = 11
# num_train = 400
num_train = 800
num_steps = 499
num_traj = 1001
num_states = 9
num_inputs = 6
dt = 0.001

file_path1 = 'Biped_files/v_0_3ms.pickle'
file_path2 = 'Biped_files/v_1ms.pickle'
with open(file_path1, 'rb') as f:
        data1=pickle.load(f)
with open(file_path2, 'rb') as f:
        data2=pickle.load(f)
        data2=np.concatenate((data2,data1),axis=0)
        data2=np.reshape(data2,(1001,500,156))
        X=data2[:,:,120:129]
        # X=np.delete(X,[9,10,11,24,25,26,27,28,29], axis=2)
        U=data2[:,:-1,150:]
print('X',X.shape,'U',U.shape)
print('check for nan values',sum(sum(sum(np.isnan(X)))),sum(sum(sum(np.isnan(U)))))

xs_train, us_train = X[:num_train,:,:],U[:num_train,:,:]
xs_val, us_val = X[num_train:,:,:],U[num_train:,:,:]
# auto encoder parameters
net_params = {}
net_params['state_dim'] = num_states
net_params['ctrl_dim'] = num_inputs
net_params['encoder_hidden_width'] = 64
net_params['encoder_hidden_depth'] = 3
net_params['encoder_output_dim'] = 6
# net_params['optimizer'] = 'sgd'
net_params['optimizer'] = 'adam'
net_params['activation_type'] = 'tanh'
net_params['lr'] = 1e-2 # 5e-4
net_params['momentum'] = 0.8
net_params['epochs'] = 50
net_params['batch_size'] = 5120
net_params['lifted_loss_penalty'] = 0.1
net_params['l2_reg'] = 0.0         
net_params['l1_reg'] = 0.0
net_params['first_obs_const'] = True
net_params['override_C'] = True # this makes the state a part of lifted state
net_params['dt'] = dt
# Koopman Model
train=True
file = 'saved_models/biped/linear/Koop_arch_9-6-64-3.pth'
if train == True:
    net = KoopmanNetCtrl_linear(net_params)
    model_koop_dnn = KoopDNN_linear(net)
    model_koop_dnn.set_datasets(xs_train, u_train=us_train, x_val=xs_val, u_val=us_val)
    model_koop_dnn.model_pipeline(net_params)
    model_koop_dnn.construct_koopman_model()
    torch.save(model_koop_dnn,file)
else:
    model_koop_dnn = torch.load(file)



train_loss = [l[0] for l in model_koop_dnn.train_loss_hist]
train_pred_loss = [l[1] for l in model_koop_dnn.train_loss_hist]
train_lifted_loss = [l[2] for l in model_koop_dnn.train_loss_hist]
val_loss = [l[0] for l in model_koop_dnn.val_loss_hist]
val_pred_loss = [l[1] for l in model_koop_dnn.val_loss_hist]
val_lifted_loss = [l[2] for l in model_koop_dnn.val_loss_hist]
epochs = np.arange(0, net_params['epochs'])

plt.figure(figsize=(15,8))
plt.plot(epochs, train_loss, color='tab:orange', label='Training loss')
plt.plot(epochs, train_pred_loss, '--', color='tab:orange', label='Training prediction loss')
plt.plot(epochs, train_lifted_loss, ':', color='tab:orange', label='Training lifted loss')
plt.plot(epochs, val_loss, color='tab:blue', label='Validation loss')
plt.plot(epochs, val_pred_loss, '--', color='tab:blue', label='Validation prediction loss')
plt.plot(epochs, val_lifted_loss, ':', color='tab:blue', label='Validation lifted loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(axis='y')
plt.yscale('log')
plt.savefig('results/biped_loss_9-6-64-3.png')
plt.show()




