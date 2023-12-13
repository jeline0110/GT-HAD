import os 
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pdb
import numpy as np 

save_dir = '../roc/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

color_list = ['b', 'c', 'g', 'lawngreen', 'k', 'm', 'pink', 'slategray', 'orange', 'y', 'r']
method_list = ['RX', 'KIFD', '2S-GLRT', 'MsRFQFT', 'CRD', 'GTVLRR', 'PTA', 'PCA-TLRSR', 'Auto-AD', 'LREN', 'GT-HAD']
file_list =  ['los-angeles-1', 'los-angeles-2', 'gulfport', 'texas-goast', 'cat-island', 'pavia']
for file in file_list:
    method_dict = {}
    mat_dir = os.path.join('../results/', file)
    for method in method_list:
        mat_name = os.path.join(mat_dir, method + '_roc.mat')
        if not os.path.exists(mat_name):
            continue
        mat = sio.loadmat(mat_name)
        tpr = mat['PD'] 
        fpr = mat['PF']
        method_dict[method] = [tpr, fpr]

    # draw roc fig
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_xscale("log", base=10)
    ax.grid(False)
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0.0, 1.05)
    plt.xlabel('False alarm rate', fontproperties='Times New Roman', 
        fontsize=15, fontweight='bold')
    plt.ylabel('Probability of detection', fontproperties='Times New Roman',
        fontsize=15, fontweight='bold')
    plt.xticks(fontproperties='Times New Roman', size=13, weight='bold')
    plt.yticks(fontproperties='Times New Roman', size=13, weight='bold')

    idx = 0
    for key in method_dict.keys():
        fpr = method_dict[key][1]
        tpr = method_dict[key][0]
        color = color_list[idx]
        idx += 1
        if len(tpr.shape) == 2 and tpr.shape[0] == 1:
            fpr = fpr[0]
            tpr = tpr[0]
        ax.semilogx(fpr, tpr, color=color, lw=3.5, label=key)
        plt.legend(loc="lower right",
            prop={"family" : "Times New Roman", 'weight': 'bold', 'size': 12})
        # cal auc
        roc_auc = auc(fpr, tpr)
        print(key + ':%.4f' % roc_auc)

    save_name = os.path.join(save_dir, file + '.pdf')
    plt.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close() 


