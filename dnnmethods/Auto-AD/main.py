# https://github.com/RSIDEA-WHU2020/Auto-AD
from __future__ import print_function
import matplotlib.pyplot as plt
# matplotlib inline

import os
import numpy as np
import time
import scipy.io
from models.skip import skip
import torch
import torch.optim
from utils.inpainting_utils import *
import shutil
import scipy.io as sio
import pdb
from sklearn.metrics import roc_auc_score, roc_curve

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../../data/'
save_dir = '../../results/'

def main(file):
    # data input
    # **************************************************************************************************************
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir, file)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    # load data
    mat = sio.loadmat(data_path)
    img_np = mat['data']
    img_np = img_np.transpose(2, 0, 1) # b, h, w
    img_np = img_np - np.min(img_np)
    img_np = img_np / np.max(img_np) # [0, 1]
    gt = mat['map']
    img_var = torch.from_numpy(img_np).type(dtype)
    band, row, col = img_var.size()
    # training paras
    thres = 0.00001
    channellss = 128
    layers = 5
    # model setup
    # **************************************************************************************************************
    pad = 'reflection' #'zero'
    OPT_OVER = 'net'
    # OPTIMIZER = 'adam'
    method = '2D'
    input_depth = img_np.shape[0]
    LR = 1e-2 # 0.01
    num_iter = 1001
    param_noise = False
    reg_noise_std = 0.1 # 0 0.01 0.03 0.05
    net = skip(input_depth, img_np.shape[0],
               num_channels_down = [channellss] * layers,
               num_channels_up =   [channellss] * layers,
               num_channels_skip =    [channellss] * layers,
               filter_size_up = 3, filter_size_down = 3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    net = net.type(dtype) # see network structure
    net_input = get_noise(input_depth, method, img_np.shape[1:]).type(dtype)
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = img_var[None, :].cuda()
    mask_var = torch.ones(1, band, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()

    def closure(iter_num, mask_varr, residual_varr):

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out_np = out.detach().cpu().squeeze().numpy()

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 100==0 and iter_num!=0:
            # weighting block
            img_var_clone = img_var.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :])
            residual_img = temp.sum(0)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            # residuals to weights
            residual_img = r_max - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):
                mask_var_clone[0, i, :] = residual_img[:]

        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone)
        total_loss.backward()
        print("iteration: %d; loss: %f" % (iter_num+1, total_loss))

        return mask_var_clone, residual_var_clone, out_np, total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    end_iter = False
    p = get_params(OPT_OVER, net, net_input)
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)

    start = time.time()
    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step()

        if j >= 1:
            index = j-int(j/50)*50
            loss_np[0][index-1] = abs(loss-loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss

        if j == num_iter-1 or end_iter == True:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()
            roc_auc = roc_auc_score(gt.flatten(), residual_np.flatten())
            print('Auc: %.4f' % roc_auc)
            # running time
            end = time.time()
            print("Runtime：%.2f" % (end - start))
            # save results
            fpr, tpr, thre = roc_curve(gt.flatten(), residual_np.flatten())
            map_path = os.path.join(save_subdir, "Auto-AD_map.mat")
            sio.savemat(map_path, {'show': residual_np})
            roc_path = os.path.join(save_subdir, "Auto-AD_roc.mat")
            sio.savemat(roc_path, {'PD': tpr, 'PF': fpr})  
            
            return

if __name__ == "__main__":
    for file in ['los-angeles-1']: 
            #     ['los-angeles-1', 'los-angeles-2', 'gulfport', 
            # 'texas-goast', 'cat-island', 'pavia']:
        main(file)