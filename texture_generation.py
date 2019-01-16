#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 00:25:57 2019

@author: antoine
"""



from __future__ import print_function
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
try :
  os.chdir('DIP/')
except :
  pass
  
import numpy as np
from models import *
import cv2

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *


#%% Paramètres 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor # choisir ce type si aucun GPU n'est disponnible 

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.
torch.manual_seed(7)


fname = 'data/texture/text1.jpeg' # chemin vers la texture à reproduire
img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_noisy_np = pil_to_np(img_noisy_pil)

# As we don't have ground truth
img_pil = img_noisy_pil
img_np = img_noisy_np

if PLOT:
    plot_image_grid([img_np], 4, 5);
    
    
#%% Mise en place du réseau et des hyperparamètres
    
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01
gamma = 1e-8 # regularizer total variation 
loss_with_TV = False

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99

num_iter = 5000  #Initial value : 2400
input_depth = 3
figsize = 5 


net = get_net(input_depth, 'texture_nets', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
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



#%% Entrainement du réseau

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
    
    global i, out_avg, psrn_noisy_last, last_net, net_input,interpolation, loss_with_TV
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
    
    if loss_with_TV :  # pour la total variation
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

    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    #print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
      #print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
      
      print('Iteration  #{} : psnr_noisy {}; psnr_gt {}; psnr_gt_sm {}'.format(i, psrn_noisy, psrn_gt, psrn_gt_sm))
      out_np = torch_to_np(out)
      plot_image_grid([np.clip(out_np, 0, 1), 
                       np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1, interpolation=interpolation)

        
    
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

#%% Génération de texture

new_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
new_output = net(new_input)
new_out_np = torch_to_np(new_output)
plot_image_grid([np.clip(new_out_np, 0, 1)], factor=9, nrow=1)

#%% affichage des courbes de psnr lors de l'entrainement

plt.figure(figsize=(6,4))
plt.plot([t for t in range(num_iter)], psrn_noisy_list, label = 'wrt to the noisy image')
plt.plot([t for t in range(num_iter)], psrn_gt_list, label = 'wrt to the original image')
plt.plot([t for t in range(num_iter)], psrn_gt_sm_list, label = 'mean outputs wrt to ground truth')

plt.title('PSNR of the network output at each epoch')
plt.legend()