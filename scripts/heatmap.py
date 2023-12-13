import os 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb
import shutil
import numpy as np 

save_dir = '../heat_map/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def heatmap(img, save_name):
    plt.figure(figsize=(7, 7))
    _plt = sns.heatmap(img, cmap='turbo', vmax=1.0, annot=False, xticklabels=False, 
        yticklabels=False, cbar=False, linewidths=0.0, rasterized=True) 
    _plt.figure.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()

method_list = ['RX', 'KIFD', '2S-GLRT', 'MsRFQFT', 'CRD', 'GTVLRR', 'PTA', 'PCA-TLRSR', 'Auto-AD', 'LREN', 'GT-HAD']
file_list =  ['los-angeles-1', 'los-angeles-2', 'gulfport', 'texas-goast', 'cat-island', 'pavia']
for file in file_list:
    mat_dir = os.path.join('../results/', file)
    for method in method_list:
        mat_name = os.path.join(mat_dir, method + '_map.mat')
        mat = sio.loadmat(mat_name)
        img = mat['show']
        # norm
        img = img - img.min()
        img = img / img.max()
        # save fig
        save_subdir = os.path.join(save_dir, file)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        save_name = os.path.join(save_subdir, method + '.pdf')
        heatmap(img, save_name)