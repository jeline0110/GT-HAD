%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
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
% param:¦Ë
% lambda = 0.5; % los-angeles-1
% lambda = 0.5; % los-angeles-2
lambda = 0.5; % gulfport 
% lambda = 0.05; % texas-goast
% lambda = 0.05; % cat-island
% lambda = 0.05; % pavia

% param:¦Â
% beta = 0.2; % los-angeles-1
% beta = 0.2; % los-angeles-2
beta = 0.2; % gulfport 
% beta = 0.2; % texas-goast
% beta = 0.2; % cat-island
% beta = 0.2; % pavia

% param:¦Ã
% gamma = 0.05; % los-angeles-1
% gamma = 0.05; % los-angeles-2
gamma = 0.05; % gulfport 
% gamma = 0.02; % texas-goast
% gamma = 0.02; % cat-island
% gamma = 0.02; % pavia

%% GTVLRR Method
disp('Running GTVLRR, Please wait...')
tic;
Dict=ConstructionD_lilu(Y,15,20);
display = false;
[X,S] = lrr_tv_manifold(Y,Dict,lambda,beta,gamma,[H,W],display);
toc

r_gtvlrr=sqrt(sum(S.^2));
r_max = max(r_gtvlrr(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_gtvlrr > tau);
  PF(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
show=reshape(r_gtvlrr,[H,W]);
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
area = sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
disp(['Auc:',num2str(area)])

save([save_dir,'GTVLRR_map.mat'],'show')
save([save_dir,'GTVLRR_roc.mat'],'PD','PF')


