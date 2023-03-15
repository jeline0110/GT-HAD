function [Ktrn, Ktst]=kernel(X_train,X_test,opt)
% This function creates kernels for training and test sets
%
% Input:    X_train/test    - data matric (features x observation)
%           opt.type        - kernel type 'rbf' or 'linear'
%           opt.scale       - width of the Gaussian kernel    
% Output:   Ktrn            - training kernel (Ntrn x Ntrn)
%           Ktst            - test kernel (Ntrn x Ntst)
% 
% ------------------------------------------------------------------------%
% Peter Mondrup Rasmussen, DTU Informatics
% email: peter.mondrup@gmail.com  homepage: www.petermondrup.com
%
% ------------------------------------------------------------------------%
% version history:
% Oct 11 2011 - first implementation.
% ------------------------------------------------------------------------%

% Licence: The code is availabel under the MIT License (MIT). See the file
% "licence.txt" for further information.


%
N = size(X_train,2);
Nt =size(X_test,2);
switch opt.type
    case 'rbf'
        X2_train=sum(X_train.*X_train,1)';
        if isfield(opt,'test') && opt.test ==1 
            dist_train = nan;
        else
            dist_train = repmat(X2_train,1,size(X_train,2)) +repmat(X2_train',size(X_train,2),1) -2*X_train'*X_train;
        end
        X2_test=sum(X_test.*X_test,1)';
        dist_test=repmat(X2_test,1,size(X_train,2)) +repmat(X2_train',size(X_test,2),1) -2*X_test'*X_train;
        dist_test=dist_test';
        Ktrn=exp(-dist_train/opt.sigma);
        Ktst=exp(-dist_test/opt.sigma);
        %
    case 'linear'
        Ktrn=X_train'*X_train;
        Ktst=X_train'*X_test;
        %
end