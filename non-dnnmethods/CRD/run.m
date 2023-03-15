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
% w_out = 15; % los-angeles-1
w_out = 17; % los-angeles-2
% w_out = 19; % gulfport 
% w_out = 9; % texas-goast
% w_out = 25; % cat-island
% w_out = 7; % pavia

% param:w_in
% w_in = 7; % los-angeles-1
w_in = 15; % los-angeles-2
% w_in = 15; % gulfport 
% w_in = 7; % texas-goast
% w_in = 15; % cat-island
% w_in = 5; % pavia

% param:¦Ë
lamda = 1e-6;

%% 3.CRD Method
disp('Running CRD, Please wait...')
tic 
show = func_CRD(hsi,w_out,w_in,lamda);
toc;
R3value = reshape(show,1,rows*cols);
[PF,PD] = perfcurve(label_value,R3value,'1') ;
area=-sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
disp(['Auc:',num2str(area)])
        
save([save_dir,'CRD_map.mat'],'show')
save([save_dir,'CRD_roc.mat'],'PD','PF')
