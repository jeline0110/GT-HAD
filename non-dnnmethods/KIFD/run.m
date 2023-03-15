%% Start
clc;
clear;
close all;
addpath(genpath('../../data/'));
key = 'los-angeles-1';
save_dir=['../../results/', key, '/'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end
addpath(genpath('./Kernel'));

%% Load data
disp(key)
input=[key,'.mat'];
load(input)
data1 = data;
row = size(data1,1);
col = size(data1,2);
data2 = NormalizeData(data1);
tic
%% Set default parameters
disp('Running KIFD, Please wait...')
zeta = 300; % param:ζ 
data_kpca = kpca(data2, 10000, zeta, 'Gaussian',1); 
data = NormalizeData(data_kpca);
data = ToVector(data);
tree_size = floor(3 * row * col /100); 
tree_num = 1000;
%% Run global iForest
s = iforest(data, tree_num, tree_size); % 1 hyperspectral data  2 number of isolation  3 trees subsample size
%% Run local iForest iteratively
img = reshape(s, row, col);
stop_flag = 0;
index = [];
num = 1;
r0 = img;
lev = graythresh(r0);   % 
while stop_flag == 0
    [r1, flag, s1, index1] = Local_iforest(r0, data, s, index, lev); 
    r0 = r1;
    s = s1;
    index = index1;
    stop_flag = flag;
    num = num + 1;
    if num > 5 
        break;
    end
end
img = zeros(row,col);
img(index1) = 1;
index = (1:row*col)';
index(index1, :) = [];
Data_d = data(:, :);
Data_d(index1,:) = [];
s_d = iforest(Data_d, tree_num, tree_size); 
r1(index) = s_d;
r2 = 10.^r1;
%% Evaluate the results
toc
show = mat2gray(r2);
[PD,PF] = roc(map(:), show(:));
area =  -sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
disp(['Auc:',num2str(area)])

save([save_dir,'KIFD_map.mat'],'show')
save([save_dir,'KIFD_roc.mat'],'PD','PF')
