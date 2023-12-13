%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
addpath(genpath('./inexact_alm_rpca'));
key = 'gulfport';
save_dir=['../../results/', key, '/'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end

%% load data
disp(key)
input=[key,'.mat'];
hsi = load(input);
DataTest = hsi.data;
mask = double(hsi.map);
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim % norm
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

%% data prepare
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';

%% Optimal Parameters       
% param:rank r
% truncate_rank = 10; % los-angeles-1
% truncate_rank = 10; % los-angeles-2
truncate_rank = 10; % gulfport 
% truncate_rank = 0; % texas-goast
% truncate_rank = 15; % cat-island
% truncate_rank = 15; % pavia

% param:μ
% mu = 0.01; % los-angeles-1
% mu = 0.01; % los-angeles-2
mu = 0.01; % gulfport 
% mu = 0.1; % texas-goast
% mu = 0.001; % cat-island
% mu = 0.0001; % pavia

%% PTA Method
disp('Running PTA, Please wait...')
tol1=1e-4; % 无用
tol2=1e-6; % ε停止迭代的条件
maxiter=400;
alphia=1.;
beta=1;
tau=1;

tic;
[X,S,area] = AD_Tensor_LILU1(DataTest,alphia,beta,tau,mu,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
toc

show=sqrt(sum(S.^2,3));
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
% auc
r_max = max(show(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (show(:)> tau)';
  PF(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
auc_area=sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
disp(['Auc:',num2str(auc_area)])

save([save_dir,'PTA_map.mat'],'show')
save([save_dir,'PTA_roc.mat'],'PD','PF')


