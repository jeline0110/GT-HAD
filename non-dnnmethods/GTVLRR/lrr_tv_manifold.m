function [X,S] = lrr_tv_manifold(Y,A,lambda,beta,gamma,im_size,display)
% low rank representation with total variation and manifold regularization
% this routine solves the following optimization problem
% min |X|_*+lambda*|S|_2,1+beta*|HX|_1,1+gamma*tr(XLX') s.t. Y = AX+S
% created on April 24, 2018

[L,N] = size(Y);
m = size(A,2);
%% construct laplacian matrix
addpath ConstructW
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 1;
W = constructW(Y',options);
Lap = diag(sum(W))-W; % laplacian matrix

maxIter = 400;
mu = 1e-4;
mu_bar = 1e10;
rho = 1.5;
% initialization
X0 = 0;
atx_A = inv(A'*A+3*eye(m));

if nargin < 7
    display = false; % true;
end

%%
% build handlers and necessary stuff
% horizontal difference operators
FDh = zeros(im_size);
FDh(1,1) = -1;
FDh(1,end) = 1;
FDh = fft2(FDh);
FDhH = conj(FDh);

% vertical difference operator
FDv = zeros(im_size);
FDv(1,1) = -1;
FDv(end,1) = 1;
FDv = fft2(FDv);
FDvH = conj(FDv);

IL = 1./( FDhH.* FDh + FDvH.* FDv + 1);

Dh = @(x) real(ifft2(fft2(x).*FDh));
DhH = @(x) real(ifft2(fft2(x).*FDhH));

Dv = @(x) real(ifft2(fft2(x).*FDv));
DvH = @(x) real(ifft2(fft2(x).*FDvH));

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------

% no intial solution supplied
if X0 == 0
    X = zeros(m, N);
end

index = 1;

% initialize V variables
V = cell(4,1);

% initialize D variables (scaled Lagrange Multipliers)
D = cell(5,1);

%  data term (always present)
V{1} = X;         % V1
D{1} = zeros(size(Y));
D{2} = zeros(m, N);  % Lagrange multipliers
index = index + 1;

% V2
V{index} = X;
D{index+1} = zeros(m, N);
index = index + 1;

%TV
% V3
V{index} = X;
D{index+1} = zeros(m, N);

% convert X into a cube
U_im = reshape(X',im_size(1), im_size(2),m);

% V4 create two images per band (horizontal and vertical differences)
V{index+1} = cell(m,2);
D{index+2} = cell(m,2);
for kk = 1:m
    % build V4 image planes
    V{index+1}{kk}{1} = Dh(U_im(:,:,kk));   % horizontal differences
    V{index+1}{kk}{2} = Dv(U_im(:,:,kk));   %   vertical differences
    % build D5 image planes
    D{index+2}{kk}{1} = zeros(im_size);   % horizontal differences
    D{index+2}{kk}{2} = zeros(im_size);   %   vertical differences
end
clear U_im;

% L1
S = sparse(L,N);

%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol = sqrt(N)*1e-5;
iter = 1;
res = inf;
while (iter <= maxIter) && (sum(abs(res)) > tol)
    
    % solve the quadratic step (all terms depending on X)
    Xi = A'*(Y-D{1}-S);
    for j = 1:3
        Xi = Xi+ V{j} - D{j+1};
    end
    X = atx_A*Xi;
    
    % Compute the Moreau proximity operators
    for j = 1:3
        % data term (V1)
        if j == 1
            temp = X + D{j+1};
            [Us,sigma,Vs] = svd(temp,'econ');
            sigma = diag(sigma);
            svp = length(find(sigma>1/mu));
            if svp >= 1
                sigma = sigma(1:svp)-1/mu;
            else
                svp = 1;
                sigma = 0;
            end
            V{j} = Us(:,1:svp)*diag(sigma)*Vs(:,1:svp)';  %singular value thresholding
        end
        
        % data term (V2)
        if j == 2
            coef = 2*gamma*Lap + mu*speye(N);
            temp2 = mu*(X+D{j+1})';
            M = (V{j})';
            parfor i = 1:m
                [M(:,i),~,~,~,~] = pcg(coef,temp2(:,i));
            end
            V{j} = M';
        end
            
        % TV  (V3 and V4)
        if  j == 3
            nu_aux = X + D{j+1};
            % convert nu_aux into image planes
            % convert X into a cube
            nu_aux5_im = reshape(nu_aux',im_size(1), im_size(2),m);
            % compute V3 in the form of image planes
            for k = 1:m
                % V3
                V3_im(:,:,k) = real(ifft2(IL.*fft2(DhH(V{j+1}{k}{1}-D{j+2}{k}{1}) ...
                    +  DvH(V{j+1}{k}{2}-D{j+2}{k}{2}) +  nu_aux5_im(:,:,k))));
                % V4
                aux_h = Dh(V3_im(:,:,k));
                aux_v = Dv(V3_im(:,:,k));
                
                V{j+1}{k}{1} = soft(aux_h + D{j+2}{k}{1}, beta/mu);   %horizontal
                V{j+1}{k}{2} = soft(aux_v + D{j+2}{k}{2}, beta/mu);   %vertical
                
                % update D5
                D{j+2}{k}{1} =  D{j+2}{k}{1} + (aux_h - V{j+1}{k}{1});
                D{j+2}{k}{2} =  D{j+2}{k}{2} + (aux_v - V{j+1}{k}{2});
            end
            % convert V3 to matrix format
            V{j} = reshape(V3_im, prod(im_size),m)';            
        end
    end
    
    S = solve_l1l2(Y-A*X-D{1},lambda/mu);
    
    % update Lagrange multipliers    
    for j = 1:4
        if  j == 1
            D{j} = D{j} - (Y-A*X-S);
        else
            D{j} = D{j} + (X-V{j-1});
        end
    end
    
    % compute residuals
    if mod(iter,10) == 1
        st = [];
        for j = 1:4
            if  j == 1
                res(j) = norm(Y-A*X-S,'fro');
                st = strcat(st,sprintf(' res(%i) = %2.6f',j,res(j) ));
            else
                res(j) = norm(X-V{j-1},'fro');
                st = strcat(st,sprintf('  res(%i) = %2.6f',j,res(j) ));
            end
        end
        if display
            fprintf(strcat(sprintf('iter = %i -',iter),st,'\n'));
        end
    end
    
    iter = iter + 1;    
    mu = min(mu*rho, mu_bar);
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end