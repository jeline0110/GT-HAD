%% Demo for 2S-GLRT
% Paper: Multi-Pixel Anomaly Detection With an Unknown Pattern for Hyperspectral Imagery
% Author: Jun Liu; Zengfu Hou ; Wei Li; Ran Tao;Danilo Orlando; Hongbin Li;
% Compiled by Zengfu Hou
% Time: 2020-06-10
% All Rights Reserved,
% Email: zephyrhours@gmail.com
%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
key = 'los-angeles-2';
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

%% Optimal Parameters       
% param:w_out
% w_out = 9; % los-angeles-1
w_out = 19; % los-angeles-2
% w_out = 13; % gulfport 
% w_out = 9; % texas-goast
% w_out = 25; % cat-island
% w_out = 21; % pavia

% param:w_in
% w_in = 7; % los-angeles-1
w_in = 15; % los-angeles-2
% w_in = 11; % gulfport 
% w_in = 5; % texas-goast
% w_in = 3; % cat-island
% w_in = 5; % pavia

%%  Proposed 2S-GLRT
disp('Running 2S-GLRT, Please wait...')

tic
show = func_2S_GLRT(hsi,w_out,w_in); 
toc;
R0value = reshape(show,1,rows*cols);
[PF,PD] = perfcurve(label_value,R0value,'1') ;
area=-sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
disp(['Auc:',num2str(area)])

save([save_dir,'2S-GLRT_map.mat'],'show')
save([save_dir,'2S-GLRT_roc.mat'],'PD','PF')


