function result = func_1S_GLRT(hsi, win_out, win_in)
%% 1S_GLRT
% Compiled by ZephyrHou on 2020-06-11
%
% Usage
%   [result] = func_2S_GLRT(hsi, win_out, win_in)
% Inputs
%   hsi - 3D data matrix (num_row x num_col x num_dim)
%   win_out - spatial size window of outer(e.g., 3, 5, 7, 9,...)
%   win_in - spatial size window of inner(e.g., 3, 5, 7, 9,...)
% Outputs
%   result - Detector output (num_row x num_col)  
%% 对整个数据进行归一化
%%% 方法一
% max_hsi=max(max(max(hsi)));
% min_hsi=min(min(min(hsi)));
% hsi=(hsi-min_hsi)/(max_hsi-min_hsi);
%%% 方法二
[~,~,bands]=size(hsi);
for i=1:bands
    
    hsi(:,:,i)=(hsi(:,:,i)-min(min(hsi(:,:,i))))/(max(max(hsi(:,:,i)))-min(min(hsi(:,:,i))));
end


%%
[rows,cols,bands] = size(hsi);
result = zeros(rows,cols);
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;

% padding avoid edges (根据窗口尺寸自适应填充边界)
DataTest = zeros(rows+2*t,cols+2*t, bands);
DataTest(t+1:rows+t, t+1:cols+t, :) = hsi;
DataTest(t+1:rows+t, 1:t, :) = hsi(:, t:-1:1, :);
DataTest(t+1:rows+t, t+cols+1:cols+2*t, :) = hsi(:, cols:-1:cols-t+1, :);
DataTest(1:t, :, :) = DataTest(2*t:-1:(t+1), :, :);
DataTest(t+rows+1:rows+2*t, :, :) = DataTest(t+rows:-1:(rows+1), :, :);

for i = t+1:cols+t 
    for j = t+1:rows+t
        block = DataTest(j-t: j+t, i-t: i+t, :);
        Xblock=DataTest(j-t1: j+t1, i-t1: i+t1, :);
        X=reshape(Xblock,win_in*win_in,bands)';   % bands x nums_in
        
        block(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;
        block = reshape(block, M, bands);
        block(isnan(block(:, 1)), :) = [];
        Y = block';  % bands x num_sam
        
        
        R_inv = pinv(Y*Y.');
        X1 =(sqrtm(R_inv))*X  ;  % bands x nums_in
        
        %% 自动进行判断，当直接执行出错后，进行特征值分解的伪逆运算操作
        try
            BB=X1*pinv((eye(size(X1,2))+X1'*X1))*X';
        catch   
            temp=(eye(size(X1,2))+X1'*X1);
            [eig_XL,eig_Z]=eig(temp);
            [Deig_Z,ind]=sort(diag(eig_Z),'descend');
            D_eigXL=eig_XL(:,ind');

            % 自动确定选择的主成分个数
            rate = 0.9999;%该参数可调   
            Sumva1 = rate * sum(Deig_Z); %按总和0.99999比例大小取舍特征值
            T0=cumsum(Deig_Z);           % cumsum为累加函数，向下累加  
            ki=find(T0>Sumva1);   
            PCs=ki(1);
            InvCov=D_eigXL(:,1:PCs)*inv(diag(Deig_Z(1:PCs)))*D_eigXL(:,1:PCs)';
            BB=X1*InvCov*X';
        end
        %%
  
        [x2,y2] = eig(BB); 
        eigenvalue2 = diag(y2);%求对角线向量
        GLRT_1S_test = max(eigenvalue2);
        
        result(j-t, i-t) = GLRT_1S_test;
    end
end
end

