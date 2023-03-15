function out = pre_image(spectraldata, No_Train, dimension, kernel, KNN, parameter)
% This function is part of the pipeline that produces the results of the
% analysis of the object recognition data set presented in the paper
% entitled "Nonlinear denoising and analysis of 
% neuroimages with kernel principal component analysis and pre-image estimation".
%
% ------------------------------------------------------------------------%
% Trine Abrahamsen, DTU Informatics
% email: tjab@imm.dtu.dk
% Peter Mondrup Rasmussen, DTU Informatics
% email: peter.mondrup@gmail.com  homepage: www.petermondrup.com
%
% ------------------------------------------------------------------------%
% version history:
% Oct 11 2011 - first implementation.
% ------------------------------------------------------------------------%

% Licence: The code is availabel under the MIT License (MIT). See the file
% "licence.txt" for further information.


if nargin <  3, error('not enough input'); end
if nargin <  4
    if strncmp(kernel,'Gaussian',1)
        parameter=1;
    elseif strncmp(kernel,'poly',1) 
        parameter=[1, 3];
    end
end
    
[nrows,ncols,nbands] = size(spectraldata);
X0 = reshape(spectraldata,nrows*ncols,nbands);
clear spectraldata

%% sub-sample for training, select No_Train samples randomly
rand('state',4711007);% initialization of rand
if No_Train>nrows*ncols, No_Train = nrows*ncols; end
idxtrain = randsample(nrows*ncols, No_Train);
Xtrain = double(X0(idxtrain,:));
ntrain = size(Xtrain,1);

Xtest = X0;
ntest = size(Xtest,1);
clear X0;

%% kernelized training data and centering the kernel matrix
[Ktrain scale Xtrainsum] = kernelize_training(kernel, Xtrain, parameter);
%meanKtrain = mean(Ktrain(:));
%meanrowsKtrain = mean(Ktrain);
Ktrain = centering(Ktrain);

%% select the first dimension eigvectors
dimout = ntrain;
if dimout>dimension, dimout=dimension; end
[eigvector,eigvalue,flagk] = eigs(Ktrain, dimout, 'LM');

if flagk~=0, warning('*** Convergence problems in eigs ***'); end
eigvalue = diag(abs(eigvalue))';
eigvector = sqrt(ntrain-1)*eigvector*diag(1./sqrt(eigvalue));
sigma=2*scale^2;
z = nan(nbands,ntest);
for i = 1:ntest
     k=kernelize_test(kernel, Xtrain, Xtest(i,:), scale, Xtrainsum);
     z(:,i) = kwokaux(eigvector',Xtrain',Ktrain',k',sigma,KNN);
end

out= reshape(z',nrows,ncols,nbands);
clear z
end
%
%% Define preimage estimation functions
%
%
function z = kwokaux(eigvector,Xtrain,Ktrain,k,sigma,KNN)
% This function estimates preimage based on Kwok and Tsang's method.
% "Kwok, J. T.-Y., Tsang, I. W.-H.
% The pre-image problem in kernel methods.
% IEEE transactions on neural networks 15 (6), 1517 1�725, 2004."
%
%
% Input:    evec        -   eigenvectors (N x K)   (scaled, 1 = nu*alpha'*alpha and sorted)
%           X           -   input point associated with each coeff [P x N]
%           K           -   Kernel matrix holding examples (N x N)
%           k           -   Kernel vector holding elements for example to be denoised (N x 1)
%           sigma       -   width of the rbf kernel
%           options.nn  -   number of neighbours
%
% ------------------------------------------------------------------------%
% Trine Abrahamsen, DTU Informatics
% email: tjab@imm.dtu.dk
% Peter Mondrup Rasmussen, DTU Informatics
% email: pmra@imm.dtu.dk
%
% ------------------------------------------------------------------------%
% version history:
% Oct 11 2011 - first implementation.
% ------------------------------------------------------------------------%



N = size(Xtrain,2);
H = eye(N)-1/N*ones(N);
M = eigvector*eigvector';
o = ones(N,1);
%
%
%% Feature space distances squared
q = H'*M*H*(k-1/N*Ktrain*o);
const = (k+1/N*Ktrain*o)'*q + 1/power(N,2)*o'*Ktrain*o + 1;
df2 = -2*Ktrain*q - 2/N*Ktrain*o + const; % eq. 9 in Kwok and Tsang's paper
%
%% Input space distance squared
d2 = real(-sigma * log( 1 - 0.5*df2));
%
%% Select nn neighbours
[~, inx] = sort(df2);
Xtrain = Xtrain(:,inx(1:KNN));
d2 = d2(inx(1:KNN));
H = eye(KNN,KNN) - 1/nn * ones(KNN,KNN);
[U,L,V] = svd(Xtrain*H,0);
r = rank(L,1e-5*max(diag(L)));
U = U(:,1:r);
L = L(1:r,1:r);
V = V(:,1:r);
Z = L*V';
d02 = sum(Z.^2)';
z = -0.5*pinv(Z')*(d2-d02);
z = U*z + sum(Xtrain,2)/KNN;
end