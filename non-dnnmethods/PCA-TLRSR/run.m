%% start
clear all;
close all;
clc;
addpath(genpath('./tSVD'));
addpath(genpath('./proximal_operator'));
addpath(genpath('../../data/')); 
key = 'los-angeles-1';
save_dir=['../../results/', key, '/'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end

%% load data
disp(key)
input=[key,'.mat'];
hsi=load(input);
DataTest = hsi.data;
mask = double(hsi.map);
% param:d
numb_dimension = 4; % los-angeles-1
% numb_dimension = 5; % los-angeles-2
% numb_dimension = 17; % gulfport 
% numb_dimension = 15; % texas-goast
% numb_dimension = 15; % cat-island
% numb_dimension = 4; % pavia

DataTest = PCA_img(DataTest, numb_dimension);
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim 
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end 

%% data process
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                        
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';
X=DataTest;  
[n1,n2,n3]=size(X);

%% ==========================Contrast Experiment==============================
%% PCA-TLRSR
disp('Running PCA-TLRSR, Please wait...')

% param:λ 
opts.lambda = 0.06; % los-angeles-1
% opts.lambda = 0.06; % los-angeles-2
% opts.lambda = 0.06; % gulfport 
% opts.lambda= 0.06; % texas-goast
% opts.lambda = 0.06; % cat-island
% opts.lambda = 0.05; % pavia

opts.mu = 1e-4;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.max_iter = 100;
opts.DEBUG = 0;

tic;
[ L,S,rank] = dictionary_learning_tlrr( X, opts);
max_iter=100;
Debug = 0;

% param:λ' 
lambda = 0.01; % los-angeles-1
% lambda = 0.01; % los-angeles-2
% lambda = 0.05; % gulfport 
% lambda= 0.02; % texas-goast
% lambda = 0.05; % cat-island
% lambda = 0.01; % pavia

[Z,tlrr_E,Z_rank,err_va ] = TLRSR(X,L,max_iter,lambda,Debug);
toc;
    
%% vis and Auc
E=reshape(tlrr_E, num, Dim)';
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
taus = linspace(0, r_max, 5000);
PF=zeros(1,5000);
PD=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area=sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
show=reshape(r_new,[H,W]);
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
disp(['Auc:',num2str(area)])

save([save_dir,'PCA-TLRSR_map.mat'],'show')
save([save_dir,'PCA-TLRSR_roc.mat'],'PD','PF')

