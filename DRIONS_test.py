'''
Author: your name
Date: 2020-08-05 16:11:32
LastEditTime: 2020-08-05 22:47:38
LastEditors: your name
Description: In User Settings Edit
FilePath: \DRIU Models\DRIU_DRIONS\DRIONS_test.py
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy
import sys
import numpy as np
import pickle
from deploy_DRIONS import DRIONSNet
import torch
import cv2
from torch import nn
from torch.nn import init
from torch.autograd import Variable


# Point to Caffe folder /path/to/caffe
caffe_root = '../'

# Choose between 'DRIVE', 'STARE', 'DRIONS', and 'RIMONE'
database = 'DRIONS'

# Use GPU?
use_gpu = 0;
gpu_id = 0;


import caffe
os.chdir(caffe_root+'/DRIU/')

def imshow_im(im):
    plt.imshow(im,interpolation='none',cmap=cm.Grays_r)
    
net_struct = 'deploy_'+database+'.prototxt'

data_root = caffe_root+'/DRIU/Images/'+database+'/'
save_root = caffe_root+'/DRIU/testresults/'+database+'/'
if not os.path.exists(save_root):
    os.makedirs(save_root)
    
with open(data_root+'test_'+database+'.txt') as f:
    imnames = f.readlines()

test_lst = [data_root+x.strip() for x in imnames]

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)


# load net
	
for idx in range(0,len(test_lst)):
    print("Scoring DRIU for image " + imnames[idx][:-1])
    
    #Read and preprocess data
    im = Image.open(test_lst[idx])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1] #BGR
    in_ -= np.array((171.0773,98.4333,58.8811)) #Mean substraction
    in_ = in_.transpose((2,0,1))
    in_ = np.expand_dims(in_, 0)

    image = np.copy(in_)
    net = DRIONSNet()
    # 载入参数
    net.load_weights_from_pkl('weights.pkl')
    # 设置为测试模式
    net.train()
    # 前向传播
    #print(torch.from_numpy(image))
    out = net(torch.from_numpy(image))
    out = out.detach().numpy()
    print(out.shape)
    print(out)
    np.save('relu.pytorch.npy', out)
    out = out[0][0, :, :]
    plt.imshow(out)
    plt.show()

  #  dconv = nn.ConvTranspose2d(in_channels=3, out_channels= 3,  kernel_size=4, stride=2, padding=1,output_padding=0, bias= False)
  #  init.constant(dconv.weight, 1)

  #  print(image)
  #  print(dconv(torch.from_numpy(image)))