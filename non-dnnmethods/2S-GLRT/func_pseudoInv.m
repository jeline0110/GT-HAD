function InvCov= func_pseudoInv(imgVec,OptPara)
%% 求矩阵的伪逆运算
% Author: Zengfu Hou
% Time: 2019-11-29
%% Function Usage
% Input:
%     imgVec -- The input matrix
%     OptPara --  Optinal parameters,the default value is 1
% Output:
%     InvCov -- The preudo inverse matrix of imgVec 
%% Main Function

if nargin==1
    OptPara = 1;
elseif OptPara~=fix(OptPara)
    error('MATLAB:func_pseudoInv:WrongNValue',...
        'The value of opt can only be a strictly positive integer 1 or 2!')
end
    

if OptPara == 1
    %% 方法一：主成分分析方法求伪逆（同方法二效果相同）
    % 此处相当于PCA变换，取前PCs个特征值与特征向量
    [eig_XL,eig_Z]=eig(imgVec);
    [Deig_Z,ind]=sort(diag(eig_Z),'descend');
    D_eigXL=eig_XL(:,ind');

    % 自动确定选择的主成分个数
    rate = 0.9999;%该参数可调   
    Sumva1 = rate * sum(Deig_Z); %按总和0.99999比例大小取舍特征值
    T0=cumsum(Deig_Z);           % cumsum为累加函数，向下累加  
    ki=find(T0>Sumva1);   
    PCs=ki(1);

    InvCov=D_eigXL(:,1:PCs)*inv(diag(Deig_Z(1:PCs)))*D_eigXL(:,1:PCs)';

  
elseif OptPara == 2
    %% 方法二：奇异值分解求伪逆(已验证，可行)
    % 此处相当于PCA变换
    [U_tmp,S_tmp,V_tmp]=svd(imgVec);    
    % PCs=size(S_tmp,2);              % S_tmp为方阵，只显示有意义的主成分数   
    rate = 0.9999;  %该参数可调
    Sumva1 = rate * sum(diag(S_tmp)); %按总和0.99999比例大小取舍特征值
    T0=cumsum(diag(S_tmp));           % cumsum为累加函数，向下累加  
    ki=find(T0>Sumva1);
    
    PCs=ki(1);
    inv_S_tmp=S_tmp(1:PCs,1:PCs);        % num_Pcs x  num_Pcs                
    inv_S_tmp=diag(ones(PCs,1)./diag(inv_S_tmp));   % num_Pcs x  num_Pcs 
    InvCov=V_tmp(:,1:PCs)*(inv_S_tmp)*U_tmp(:,1:PCs)'; % num_Pcs x  num_Pcs
end

end

