
# 6-128-3,reduced input state dimension to 21, removed scaling in loss function, optim adam, Koop_arch2_6-128-3.pth
# lr schedule, increased batch size, increased epochs, grad cliping

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

num_train = 800
num_steps = 499
num_traj = 1001
num_states = 21
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
        X=data2[:,:,120:150]
        X=np.delete(X,[9,10,11,24,25,26,27,28,29], axis=2)
        U=data2[:,:-1,150:]

print('X',X.shape,'U',U.shape)
print('check for nan values',sum(sum(sum(np.isnan(X)))),sum(sum(sum(np.isnan(U)))))

xs_train, us_train = X[:num_train,:,:],U[:num_train,:,:]
xs_val, us_val = X[num_train:,:,:],U[num_train:,:,:]

bilinear = True
# auto encoder parameters
net_params_bilin = {}
net_params_bilin['state_dim'] = num_states
net_params_bilin['ctrl_dim'] = num_inputs
net_params_bilin['encoder_hidden_width'] = 128
net_params_bilin['encoder_hidden_depth'] = 3
net_params_bilin['encoder_output_dim'] = 6
net_params_bilin['optimizer'] = 'adam'
net_params_bilin['activation_type'] = 'tanh'
net_params_bilin['lr'] = 1e-3 # 5e-4
net_params_bilin['epochs'] = 50
net_params_bilin['batch_size'] = 1280
net_params_bilin['lifted_loss_penalty'] = 0.2
net_params_bilin['l2_reg'] = 1e-5
net_params_bilin['l1_reg'] = 1e-5
net_params_bilin['first_obs_const'] = True
net_params_bilin['override_C'] = True # this makes the state a part of lifted state
net_params_bilin['dt'] = dt

train = True
standardize = False
file_koop_bilinear = 'saved_models/biped/bilinear/Koop_arch_6-128-3.pth'

if train == True:
    if standardize:
        standardizer_u_kdnn = fit_standardizer(us_train, preprocessing.StandardScaler(with_mean = True))
        standardizer_x_kdnn = fit_standardizer(xs_train, preprocessing.StandardScaler(with_mean = False))
        net = KoopmanNetCtrl(net_params_bilin, standardizer_x=standardizer_x_kdnn, standardizer_u=standardizer_u_kdnn)
    else:
        net = KoopmanNetCtrl(net_params_bilin)

    # pre_trained_model = torch.load(file_koop_bilinear)
    # net.load_state_dict(pre_trained_model)
    model_koop_dnn_bilin = KoopDNN(net)
    model_koop_dnn_bilin.set_datasets(xs_train, u_train=us_train, x_val=xs_val, u_val=us_val)
    model_koop_dnn_bilin.model_pipeline(net_params_bilin, early_stop= False)
    model_koop_dnn_bilin.construct_koopman_model()
    torch.save(model_koop_dnn_bilin,file_koop_bilinear)
else:
    model_koop_dnn_bilin = torch.load(file_koop_bilinear)

train_loss = [l[0] for l in model_koop_dnn_bilin.train_loss_hist]
train_pred_loss = [l[1] for l in model_koop_dnn_bilin.train_loss_hist]
train_lifted_loss = [l[2] for l in model_koop_dnn_bilin.train_loss_hist]
val_loss = [l[0] for l in model_koop_dnn_bilin.val_loss_hist]
val_pred_loss = [l[1] for l in model_koop_dnn_bilin.val_loss_hist]
val_lifted_loss = [l[2] for l in model_koop_dnn_bilin.val_loss_hist]
epochs = np.arange(0, net_params_bilin['epochs'])

plt.figure(figsize=(15,8))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(epochs, train_loss, color='tab:orange', label='Training loss')
plt.plot(epochs, train_pred_loss, '--', color='tab:orange', label='Training prediction loss')
# plt.plot(epochs, train_lifted_loss, ':', color='tab:orange', label='Training lifted loss')
plt.plot(epochs, val_loss, color='tab:blue', label='Validation loss')
plt.plot(epochs, val_pred_loss, '--', color='tab:blue', label='Validation prediction loss')
# plt.plot(epochs, val_lifted_loss, ':', color='tab:blue', label='Validation lifted loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Bilinear Model loss 6-128-3')
plt.grid(axis='y')
plt.savefig('results/Bilinear/biped_6-128-3.png')
plt.show()





