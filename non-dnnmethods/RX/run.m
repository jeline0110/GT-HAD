%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
key = 'los-angeles-1';
save_dir=['../../results/', key, '/'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end

%% Load HSI dataset
disp(key)
input=[key,'.mat'];
data = load(input);
hsi = data.data;
mask = double(data.map);
[rows,cols,bands]=size(hsi);
label_value=reshape(mask,1,rows*cols);

%% RX Method
disp('Running RX, Please wait...')
tic
show = func_RX(hsi);
toc;
R1value = reshape(show,1,rows*cols);
[PF,PD] = perfcurve(label_value,R1value,'1') ;
area=-sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
disp(['Auc:',num2str(area)])

save([save_dir,'RX_map.mat'],'show')
save([save_dir,'RX_roc.mat'],'PD','PF')
