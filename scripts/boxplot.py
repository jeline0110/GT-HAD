import matplotlib.pyplot as plt
import os 
import scipy.io as sio
import pdb
import cv2
import shutil
import numpy as np 

gt_dir = '../data/'
save_dir = '../box_plot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def del_zero(x):
    tmp = []
    for i in range(x.shape[0]):
        if x[i] != 0:
            tmp.append(x[i])
    y = np.array(tmp)

    return y

method_list = ['RX', 'KIFD', '2S-GLRT', 'MsRFQFT', 'CRD', 'GTVLRR', 'PTA', 'PCA-TLRSR', 'Auto-AD', 'LREN', 'GT-HAD']
file_list =  ['los-angeles-1', 'los-angeles-2', 'gulfport', 'texas-goast', 'cat-island', 'pavia']
for file in file_list:
    gt_path = os.path.join(gt_dir, file + '.mat')
    mat = sio.loadmat(gt_path)
    gt = mat['map']

    mat_dir = os.path.join('../results/', file)
    data = []
    for method in method_list:
        mat_name = os.path.join(mat_dir, method + '_map.mat')
        mat = sio.loadmat(mat_name)
        img = mat['show']
        # norm
        img = img - img.min()
        img = img / img.max()
        bk = img.copy()
        ab = img.copy()
        # data process
        bk[gt != 0] = 0
        bk = bk.flatten()
        bk = del_zero(bk)
        ab[gt == 0] = 0
        ab = ab.flatten()
        ab = del_zero(ab)
        data.append(ab)
        data.append(bk)

    # draw boxplot fig
    ax = plt.subplot()
    ax.grid(False)
    ax.set_ylim(0.0, 1.19)
    plt.ylabel('Normalized detection statistic range', fontproperties='Times New Roman',
        fontsize=15, fontweight='bold')
    plt.yticks(fontproperties='Times New Roman', size=12, weight='bold')

    num = 9
    # red & blue box
    color_list = [(1, 0, 0), (0, 0, 1)] * num
    box_position = []
    method_position = []
    pos = 0.0
    for i in range(num * 2):
        if i % 2 == 0:
            pos += 1.0
            method_position.append(pos + 0.2) # 
        else:
            pos += 0.4
        box_position.append(pos)

    bp = ax.boxplot(data, widths=0.3, patch_artist=True, showfliers=False,
            positions=box_position, medianprops={'color':'black'}, 
            whiskerprops={'linestyle':'--'})

    # set boxes
    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set(linewidth=0.75)
    # set whiskers
    color_list_double = [(1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1)] * num
    for patch, color in zip(bp['whiskers'], color_list_double):
        patch.set(color=color, linewidth=2.5)
    # set caps
    for patch, color in zip(bp['caps'], color_list_double):
        patch.set(color=color, linewidth=2.5)
    # set medians
    for patch, color in zip(bp['medians'], color_list):
        patch.set(color=color, linewidth=0.1)

    plt.xticks(method_position, method_list)
    plt.xticks(fontproperties='Times New Roman', rotation=18, size=11, weight='bold')
    labels = ["Anomaly", "Background"]
    plt.legend(bp['boxes'], labels, loc='upper right',
        prop={"family" : "Times New Roman", 'weight': 'bold', 'size': 12})
    # save fig
    save_path = os.path.join(save_dir, file + '.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.0) # 切割多余的空白区域
    plt.close()
    