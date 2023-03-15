function result = func_CRD(Data, win_out, win_in, lambda)
%% Collaborative Representation for Hyperspectral Anomaly Detector
% Compiled by Zephyr Hou on 2017-02-26
% Reference:
%        《 Collaborative Representation for Hyperspectral Anomaly Detector》
% 
%% Function Usage:
%   [result] = CRD_Detect(Data, window, lambda)
% Inputs
%   Data - 3D data matrix (num_row x num_col x num_dim)
%   window - spatial size window (e.g., 3, 5, 7, 9,...)
%   lambda - regularization parameter
% Outputs
%   result - Detector output (num_row x num_col)
%%  Main Function
[rows,cols,bands] = size(Data);        
result = zeros(rows, cols);
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;
num_sam=win_out*win_out-win_in*win_in;
%% expanding the edges(two methods)
% padding avoid edges
DataTest = zeros(3*rows, 3*cols, bands);
DataTest(rows+1:2*rows, cols+1:2*cols, :) = Data;
DataTest(rows+1:2*rows, 1:cols, :) = Data(:, cols:-1:1, :);
DataTest(rows+1:2*rows, 2*cols+1:3*cols, :) = Data(:, cols:-1:1, :);
DataTest(1:rows, :, :) = DataTest(2*rows:-1:(rows+1), :, :);
DataTest(2*rows+1:3*rows, :, :) = DataTest(2*rows:-1:(rows+1), :, :);

% padding zeros to avoid edges
% DataTest = zeros(3*rows, 3*cols, bands);
% DataTest(rows+1:2*rows, cols+1:2*cols, :) = Data;
% DataTest(rows+1:2*rows, 1:cols, :) = zeros(rows,cols,bands);
% DataTest(rows+1:2*rows, 2*cols+1:3*cols, :) = zeros(rows,cols,bands);
% DataTest(1:rows,:,:)=zeros(rows,3*cols,bands);
% DataTest(2*rows+1:3*rows,:,:)=zeros(rows,3*cols,bands);

Gamma=zeros(1,num_sam);
for i = 1+cols: 2*cols 
    for j = 1+rows: 2*rows
        block = DataTest(j-t: j+t, i-t: i+t, :);
        y = squeeze(DataTest(j, i, :)).';% 1 x num_dim
        block(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;
        block = reshape(block, M, bands); % M x bands
        block(isnan(block(:, 1)), :) = []; 
        Xs = block';  % num_dim x num_sam   
        
       %% Based on the folumate 8 （经常会出现无法求逆线性）
%         y_1=[y,1];       % 1 x (num_dim + 1 )
%         Xs_1=[Xs;ones(1,num_sam)];  % (num_dim + 1) x num_sam 
%         for k=1:num_sam
%             Gamma(1,k)=norm((y'-Xs(:,k)),2);
%         end   
%         Gamma_y=diag(Gamma);  % num_sam x num_sam       
%         weights=pinv(Xs_1'*Xs_1+lambda*(Gamma_y'*Gamma_y))*Xs_1'*y_1'; % num_sam x 1  (formula 8)      
%         y_hat = (Xs*weights(:))';  % 1 x num_dim
%         result(j-rows, i-cols) = norm(y - y_hat, 2);      
               
       %% Based on the folumate 6  
	    for k=1:num_sam
            Gamma(1,k)=norm((y'-Xs(:,k)),2);
        end
		Gamma_y=diag(Gamma);  % num_sam x num_sam 
	   
%         weights=pinv(Xs'*Xs+lambda*(Gamma_y'*Gamma_y))*Xs'*y';   % num_sam x 1(formula 6)
        %利用求伪逆代替。否则不收敛
       weights=func_pseudoInv(Xs'*Xs+lambda*(Gamma_y'*Gamma_y))*Xs'*y';
             
        y_hat = (Xs*weights(:))';  % 1 x num_dim
        result(j-rows, i-cols) = norm(y - y_hat, 2);

		
    end
end

