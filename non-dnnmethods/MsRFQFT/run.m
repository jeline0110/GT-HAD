%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
addpath('.\main_functions\')
key = 'gulfport';
save_dir=['../../results/', key, '/'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end

%% load data
disp(key)
input=[key,'.mat'];
hsi = load(input);
data=hsi.data;
map=hsi.map;

%% Optimal Parameters       
% param:sigma
% sigma = 1.2; % los-angeles-1
% sigma = 4.0; % los-angeles-2
sigma = 10; % gulfport 
% sigma = 1.4; % texas-goast
% sigma = 0.4; % cat-island
% sigma = 1.6; % pavia

%% MsRFQFT Method
disp('Running MsRFQFT, Please wait...')
result = MsRFQFT(data,sigma);

%% Result evaluation
[auc_pdpf,auc_pdtau,auc_pftau,PD,PF] =  AUCall(result,map);
disp(['Auc:',num2str(auc_pdpf)])

show=result;
save([save_dir,'MsRFQFT_map.mat'],'show')
save([save_dir,'MsRFQFT_roc.mat'],'PD','PF')
