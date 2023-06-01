# https://github.com/jeline0110/GT-AD
import os
import numpy as np
import torch
import torch.optim as Optim
import scipy.io as sio
import pdb
from net import Net
from sklearn.metrics import roc_auc_score, roc_curve
import shutil
from utils import get_params, img2mask, seed_dict
import random
from progress.bar import Bar
import time 
import torch.nn as nn 
from torch.utils.data import DataLoader
from data import DatasetHsi
from block import Block_fold, Block_search
# import cv2 

dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../../data/'
save_dir = '../../results/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(file):
    # set random seed 
    # **************************************************************************************************************
    seed = seed_dict[file]
    set_seed(seed)
    # data process
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
    img_var = img_var[None, :]
    # set block functions and init dataloader
    # **************************************************************************************************************
    patch_size = 3 
    patch_stride = 3
    block_size = patch_size * patch_stride # block_size is the sliding window size
    data_set = DatasetHsi(img_var, wsize=block_size, wstride=3)
    block_fold = Block_fold(wsize=block_size, wstride=3)
    block_search = Block_search(img_var, wsize=block_size, wstride=3)
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True, drop_last=False)
    # model setup
    # **************************************************************************************************************
    # net
    net = Net(in_chans=band, embed_dim=64, patch_size=patch_size, 
        patch_stride=patch_stride, mlp_ratio=2.0, attn_drop=0.0, drop=0.0)
    net = net.cuda()
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    # loss
    mse = torch.nn.MSELoss().type(dtype)
    # optim
    LR = 1e-3 # 2e-5
    p = get_params(net)
    optimizer = Optim.Adam(p, lr=LR)
    print('Starting optimization with ADAM')
    # train
    # **************************************************************************************************************
    end_iter = 150
    search_iter = 25 # [50, 100, 125]
    bar = Bar('Processing', max=end_iter)
    data_num = data_set.__len__()
    match_vec = torch.zeros((data_num)).type(dtype)
    search_matrix = torch.zeros((data_num, band, block_size, block_size)).type(dtype)
    search_index = torch.arange(0, data_num).type(torch.cuda.LongTensor)
    avgpool = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)) # k,s,p o=(i-k+2p)/s+1 
    # start train
    start = time.time()
    for iter in range(1, end_iter + 1):
        search_flag = True if iter % search_iter == 0 and iter != end_iter else False
        for idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            # input -> net -> output
            net_gt, net_input, block_idx = batch_data['block_gt'], batch_data['block_input'], batch_data['index'].cuda()
            net_out = net(net_input, block_idx=block_idx, match_vec=match_vec)
            if search_flag: search_matrix[block_idx] = net_out
            # cal loss
            loss = mse(net_out, net_gt)
            loss.backward()
            optimizer.step()
        
        # H-BMM
        if search_flag:
            match_vec = torch.zeros((data_num)).type(dtype) # reset match_vec
            search_back = block_fold(search_matrix.detach(), data_set.padding, row, col)
            match_vec = block_search(search_back.detach(), match_vec, search_index)
        bar.next()

        # start test 
        if iter == end_iter:
            bar.finish()
            infer_loader = DataLoader(data_set, batch_size=64, shuffle=False, drop_last=False)
            net = net.eval()
            infer_res_list = []

            for idx, data in enumerate(infer_loader):
                infer_in = data['block_input']
                infer_idx = data['index'].cuda()
                # inference  
                infer_out = net(infer_in, block_idx=infer_idx, match_vec=match_vec)
                infer_res = torch.abs(infer_in - infer_out) ** 2
                infer_res = avgpool(infer_res)
                infer_res_list.append(infer_res)

            infer_res_out = torch.cat(infer_res_list, dim=0)
            infer_res_back = block_fold(infer_res_out.detach(), data_set.padding, row, col)
            residual_np = img2mask(infer_res_back)
            # cal auc
            auc = roc_auc_score(gt.flatten(), residual_np.flatten())
            print('Auc: %.4f' % auc)
            # running time
            end = time.time()
            print("Runtime：%.2f" % (end - start)) 
            # save results
            fpr, tpr, thre = roc_curve(gt.flatten(), residual_np.flatten())
            map_path = os.path.join(save_subdir, "GT-AD_map.mat")
            sio.savemat(map_path, {'show': residual_np})
            roc_path = os.path.join(save_subdir, "GT-AD_roc.mat")
            sio.savemat(roc_path, {'PD': tpr, 'PF': fpr})

            return

if __name__ == "__main__":
    for file in ['los-angeles-1']: 
            #     ['los-angeles-1', 'los-angeles-2', 'gulfport', 
            # 'texas-goast', 'cat-island', 'pavia']:
        main(file)

