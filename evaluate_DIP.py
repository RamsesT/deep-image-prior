#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:06:40 2019

@author: antoine
"""

from __future__ import print_function
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
  
import numpy as np
from models import *
import cv2

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *
if torch.cuda.is_available() :
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor
else : 
    dtype = torch.FloatTensor
    
#dtype = torch.FloatTensor

imsize =-1
PLOT = True


def evaluate_net_arch(fname,num_iter = 3000,loss_with_TV = False,LR = 0.01,
                       OPT_OVER = 'net', max_depth = 5, start_with_blur = False,
                       input_depth=32,back_track_range=100, overfit=False, gamma = 1e-8, sigma = 25) : 
    
    
    sigma_ = sigma/255.
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)   
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    list_of_list_psnr = []
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
     # 'net,input'
    
    reg_noise_std = 1./30. # set to 1./20. for sigma=50
     # regularizer total variation 
    
    
    OPTIMIZER='adam' # 'LBFGS'
    show_every = back_track_range 
    exp_weight=0.99
    if overfit : 
        num_iter = 8000
        
     # initial value 3000
    if start_with_blur : 
      input_depth = 3
      blur_size=3
      n_c = 3
      #blur_weights = torch.ones((n_channels,n_channels,blur_size,blur_size))*1/(blur_size**2)
      net_input = np_to_torch(np.copy(img_noisy_np))
      print(net_input.shape)
      print(net_input[0].numpy().transpose(1, 2, 0).transpose(2, 0, 1).shape)
      blur_img = cv2.blur(src=net_input[0].numpy().transpose(1, 2, 0),ksize=(7,7))
      net_input_np = blur_img.transpose(2,0,1)
      net_input = torch.tensor(np_to_torch(net_input_np))
      net_input = net_input.to('cuda')

    
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=max_depth,
                  upsample_mode='bilinear').type(dtype)


   
        
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    
    print(net_input.shape)
    print(img_np.shape)
    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)
    # Loss
    
    mse = torch.nn.MSELoss().type(dtype)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    
    
    #### My work
    
      
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    interpolation = None
    
    psrn_noisy_list = [] 
    psrn_gt_list    = []
    psrn_gt_sm_list = []
    
    i = 0
    def closure():
        
        nonlocal i, out_avg, psrn_noisy_last, last_net, net_input,interpolation, loss_with_TV
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
        
        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
        
        if loss_with_TV : 
          #tv_reg = lambda y: (torch.sum(torch.abs(out[:, :, :, :-1] - out[:, :, :, 1:])) + torch.sum(torch.abs(out[:, :, :-1, :] - out[:, :, 1:, :])))
          TV_loss = gamma * (
                    torch.sum(torch.abs(out[:, :, :, :-1] - out[:, :, :, 1:])) + 
                    torch.sum(torch.abs(out[:, :, :-1, :] - out[:, :, 1:, :])))
          total_loss = mse(out, img_noisy_torch) + TV_loss
    
        else : 
          total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
            
        
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
        
        psrn_noisy_list.append(psrn_noisy) 
        psrn_gt_list.append(psrn_gt) 
        psrn_gt_sm_list.append(psrn_gt_sm) 
    
        
        
        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')
    
                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.detach().copy_(new_param.cuda())
    
                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
                
        i += 1
    
        return total_loss
    
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    return psrn_gt_list, psrn_noisy_list

def evaluate_denoising_net_depth(fname,depth_range=[1,3,5,10]):
    
    list_of_gt_list = []
    list_of_noisy_list = []
    
    for i in depth_range:
        psrn_gt_list, psrn_noisy_list = evaluate_net_arch(fname=fname,max_depth=i, overfit=True)
        list_of_gt_list.append(psrn_gt_list)
        list_of_noisy_list.append(psrn_noisy_list)
    
    return list_of_gt_list, list_of_noisy_list


def evaluate_denoising_start_with_blur(fname,sigma = 25):
    
    psrn_gt_list, psrn_noisy_list = evaluate_net_arch(fname=fname,start_with_blur=False, sigma=sigma)
    psrn_gt_list_blur, psrn_noisy_list_blur = evaluate_net_arch(fname=fname,start_with_blur=True, sigma=sigma)

    list_of_noisy_list = [psrn_noisy_list,psrn_noisy_list_blur]
    list_of_gt_list = [psrn_gt_list,psrn_gt_list_blur]
    
    return list_of_gt_list, list_of_noisy_list


def evaluate_denoising_TV(fname, gamma_list=[1e-6,1e-9,1e-8]):
    
    list_of_gt_list = []
    list_of_noisy_list = []
    LR = 0.001
    psrn_gt_list, psrn_noisy_list = evaluate_net_arch(fname=fname, LR=LR)
    list_of_gt_list = [psrn_gt_list]
    list_of_noisy_list = [psrn_noisy_list]
    for gamma in gamma_list: 
        psrn_gt_list_TV, psrn_noisy_list_TV = evaluate_net_arch(fname=fname,LR=LR,loss_with_TV=True,gamma=gamma)
        list_of_noisy_list.append(psrn_noisy_list_TV)
        list_of_gt_list.append(psrn_gt_list_TV) 
        
    return list_of_gt_list, list_of_noisy_list

#%%

#fname = 'data/denoising/F16_GT.png'
#evaluate_denoising_net_depth(fname=fname)