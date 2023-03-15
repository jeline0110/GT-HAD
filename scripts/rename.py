# 将results里的mat重命名，生成results_github，配合开源代码使用。
import os 
import pdb
import scipy.io as sio
import shutil

# rename
for root, dirs, files in os.walk('../../code_fix/results/'):
    for file in files:
        if file == '.DS_Store':
            continue
        if 'RPCA_RX' in file:
            continue
        if 'LRX' in file:
            continue  

        file_path = os.path.join(root, file)
        sub_dir = root[-2:]
        if sub_dir == 'A1':
            new_dir = 'los-angeles-1'
        elif sub_dir == 'A2':
            new_dir = 'los-angeles-2'
        elif sub_dir == 'A4':
            new_dir = 'gulfport'
        elif sub_dir == 'U1':
            new_dir = 'texas-goast'
        elif sub_dir == 'B1':
            new_dir = 'cat-island'
        elif sub_dir == 'B4':
            new_dir = 'pavia'
        else:
            continue

        save_subdir = os.path.join('../results', new_dir)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)

        mat = sio.loadmat(file_path)
        method_name = file[:-8]
        if method_name == 'GRX':
            new_name = 'RX'
        elif method_name == 'GLRT':
            new_name = '2S-GLRT'
        elif method_name == 'PCA_TLRSR':
            new_name = 'PCA-TLRSR'
        elif method_name == 'Auto_AD':
            new_name = 'Auto-AD'
        elif method_name == 'Ours':
            new_name = 'GT-AD'
        else:
            new_name = method_name

        if 'roc' in file:
            tpr = mat['PD_' + method_name.lower()]
            fpr = mat['PF_' + method_name.lower()]
            roc_path = os.path.join(save_subdir, new_name + "_roc.mat")
            sio.savemat(roc_path, {'PD': tpr, 'PF': fpr})

        if 'png' in file:
            img = mat['f_show']
            map_path = os.path.join(save_subdir, new_name + "_map.mat")
            sio.savemat(map_path, {'show': img})
