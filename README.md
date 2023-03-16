# GT-AD

This is an official implementation of GT-AD: Gated Transformer for Hyperspectral Anomaly Detection.

Framework of GT-AD:

<img src="framework.png" width=600 height=375>

### 1. Comparison Methods:

In addition to GT-AD, this repo includes the implementation of the following anomaly detection methods. DNN-based methods (Auto-AD, LREN) are available in `GT-AD/dnnmethods`, and non-DNN methods (RX, KIFD, 2S-GLRT, CRD, GTVLRR, PCA-TLRSR) are available in `GT-AD/non-dnnmethods`.

<details open>
<summary><b>Supported Algorithms:</b></summary>

* [x] [RX](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=60107)
* [x] [KIFD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8833502)
* [x] [2S-GLRT](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9404853)
* [x] [CRD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6876207)
* [x] [GTVLRR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8833518)
* [x] [PCA-TLRSR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9781337)  [![](https://img.shields.io/badge/-Github-blue)](https://github.com/MinghuaWang123/PCA-TLRSR)
* [x] [Auto-AD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9382262) [![](https://img.shields.io/badge/-Github-blue)](https://github.com/RSIDEA-WHU2020/Auto-AD)
* [x] [LREN](https://ojs.aaai.org/index.php/AAAI/article/view/16536)  [![](https://img.shields.io/badge/-Github-blue)](https://github.com/xdjiangkai/LREN)

</details>
  
Besides, we also provide their original codes in `GT-AD/original-codes`.
- RX, CRD, and 2S-GLRT are available in `GT-AD/original-codes/2S-GLRT.zip`. 
- KIFD is available in `GT-AD/original-codes/KIFD.zip`.
- GTVLRR is available in `GT-AD/original-codes/PTA.zip`.
- PCA-TLRSR is available in `GT-AD/original-codes/PCA-TLRSRT.zip`.
- Auto-AD is available in `GT-AD/original-codes/Auto-AD.zip`.
- LREN is available in `GT-AD/original-codes/LREN.zip`.

### 2. Create Environment:
#### 2.1 DNN-based Methods:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Tensorflow for LREN
- Pytorch for Auto-AD and GT-AD
- Numpy
- Sklearn
- Scipy
- Progressbar

#### 2.1 non-DNN Methods:

- MATLAB

#### 2.3 Other Requirements:

- Matplotlib
- Seaborn

### 3. Prepare Dataset:

Datasets are available in `GT-AD/data`.

### 4. Running Experiments:
#### 4.1 DNN-based Methods:
```shell
# Auto-AD
cd GT-AD/dnnmethods/Auto-AD/
python main.py 

# LREN
cd GT-AD/dnnmethods/LREN/
python main.py 

# GT-AD
cd GT-AD/dnnmethods/GT-AD/
python main.py 
```

#### 4.2 non-DNN Methods:
```shell
# RX
locate GT-AD/non-dnnmethods/RX/
run run.m 

# KIFD
locate GT-AD/non-dnnmethods/KIFD/
run run.m 

# 2S-GLRT
locate GT-AD/non-dnnmethods/2S-GLRT/
run run.m 

# CRD
locate GT-AD/non-dnnmethods/CRD/
run run.m

# GTVLRR
locate GT-AD/non-dnnmethods/GTVLRR/
run run.m

# PCA-TLRSR
locate GT-AD/non-dnnmethods/PCA-TLRSR/
run run.m
```
