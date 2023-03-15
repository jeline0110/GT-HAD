function [Z, E] =   sparse_graph_LRR(X, W, lambda, garma, beta, rho, DEBUG)
% This matlab code implements linearized ADM method for LRR problem
%------------------------------
% min |Z|_*+  lambda * |Z|_1 +  garma  *|E|_1+   beta * tr(Z* L*Z^T)
% s.t., X = XZ+E
%--------------------------------
% inputs:
%        X -- D*N data matrix
%        W -- affinity graph matrix
% outputs:
%        Z -- N*N representation matrix
%        E -- D*N sparse error matrix
%        relChgs --- relative changes
%        recErrs --- reconstruction errors
%
% created by Ming Yin, School of Automation, GDUT, China% 
% 2013-6-18
%

% If you use this code, please cite our work
% @ARTICLE{7172559, 
% author={Yin, M. and Gao, J. and Lin, Z.}, 
% journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on}, 
% title={Laplacian Regularized Low-Rank Representation and Its Applications}, 
% year={2016}, 
% volume={38}, 
% number={3}, 
% pages={504-517}, 
% doi={10.1109/TPAMI.2015.2462360}, 
% ISSN={0162-8828}, 
% month={March},}


clear global;
global M;           %  

addpath('..\utilities\PROPACK');

if (~exist('DEBUG','var'))
    DEBUG = 0;
end

if nargin < 6
    rho = 1.9;
end

if nargin < 5
    beta = 1.1;
end

if nargin < 4
    garma = 1.9;
end

if nargin < 3
    lambda = 0.1;
end

% Construct the K-NN Graph
if nargin < 2  ||  isempty(W)
     W = constructW (X');
end
DCol = full(sum(W,2));

% unnormalized Laplacian;
D = spdiags(DCol,0,speye(size(W,1)));
L = D - W; 

normfX = norm(X,'fro');
tol1 = 1e-4;              % threshold for the error in constraint
tol2 = 1e-5;              %  threshold for the change in the solutions
[d n] = size(X);
opt.tol = tol2;            %  precision for computing the partial SVD
opt.p0 = ones(n,1);

maxIter = 500;

max_mu = 1e10;
norm2X = norm(X,2);
% mu = 1e2*tol2;
mu = min(d,n)*tol2;

eta = norm2X*norm2X*1.02;   %eta needs to be larger than ||X||_2^2, but need not be too large.

%% Initializing optimization variables
% intializing
E = sparse(d,n);
Y1 = zeros(d,n);
Y2 = zeros(n,n);   
Z = eye(n, n);
J = zeros(n, n);

XZ = zeros(d, n);   

sv = 5;
svp = sv;

%% Start main loop
convergenced = 0;
iter = 0;

if DEBUG
    disp(['initial,rank(Z)=' num2str(rank(Z))]);
end

while iter<maxIter
    iter = iter + 1;
    
    %copy E, J  and Z to compute the change in the solutions
    Ek = E;
    Zk = Z;
    Jk = J;
    
    XZ = X*Z;
    ZLT = Z* L';
    ZL = Z*L;
    
    %solving Z    
    %-----------Using PROPACK--------------%
    M =  beta* (ZLT + ZL);
    M = M + mu *X' *(XZ -X + E -Y1/mu); 
    M = M +mu *(Z- J+Y2/ mu); 
    M = Z - M/eta;
    
%    [U, S, V] = lansvd(M, n, n, sv, 'L', opt);
    %[U, S, V] = lansvd(M, n, n, sv, 'L');
    [U, S, V] = svd((M+eps),'econ');
      
    S = diag(S);
    svp = length(find(S>1/(mu*eta)));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    if svp>=1
        S = S(1:svp)-1/(mu*eta);
    else
        svp = 1;
        S = 0;
    end

    A.U = U(:, 1:svp);
    A.s = S;
    A.V = V(:, 1:svp);
    
    Z = A.U*diag(A.s)*A.V';
    XZ = X*Z;               %introducing XZ to avoid computing X*Z multiple times, which has O(n^3) complexity.
    
    % solving J  
    temp = Z+Y2/mu;
    J = max(0, temp - lambda/mu) + min(0, temp + lambda/mu); 
    J = max(0,J);
    
     % solving E
    temp = X- XZ;
    temp = temp+Y1/mu;
    E = max(0, temp - garma/mu)+ min(0, temp + garma/mu);

    relChgZ = norm(Zk - Z,'fro')/normfX;
    relChgE = norm(E - Ek,'fro')/normfX;
    relChgJ = norm(J - Jk,'fro')/normfX;
    relChg =   max( max(relChgZ, relChgE), relChgJ);
    
    dY1 = X - XZ - E;
    recErr1 = norm(dY1,'fro')/normfX;    
    dY2 =  Z - J;
    recErr2 = norm(dY2,'fro')/normfX;
    recErr = max(recErr1, recErr2);
    
    convergenced = recErr <tol1  && relChg < tol2;
    
    if DEBUG
        if iter==1 || mod(iter,50)==0 || convergenced
            disp(['iter ' num2str(iter) ',mu=' num2str(mu) ...
                ',rank(Z)=' num2str(svp) ',relChg=' num2str(relChg)...
                ',recErr=' num2str(recErr)]);
        end
    end
    
    if convergenced
        break;
    else
        Y1 = Y1 + mu*dY1;
        Y2 = Y2 + mu*dY2;
        
        if mu*relChg < tol2
            mu = min(max_mu, mu*rho);
        end
    end
end