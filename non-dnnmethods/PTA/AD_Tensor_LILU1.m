function [X,S,area0] = AD_Tensor_LILU1(Y,alphia,beta,tau,mu,truncate_rank,maxIter,tol1,tol2,normal_map,anomaly_map)
%% argmin f(S,X)=tau/2*(||DhX1||_F^2+||DwX2||_F^2)+alphia*||X3||_tnnr+beta*||S3||_2,1   s.t. Y=X+S;
%% argmin f(V1,V2,V3,V4,S,X,A,B,D1,D2,D3,D4,D5,mu)=tau/2*(||DhX1||_F^2+||DwX2||_F^2)+alphia*(||X3||_*-A*X3*B'+beta*||S3||_2,1+0.5*mu(||Y-X-S+D1||_F^2+||V1-X+D2||_F^2+||V2-X+D3||_F^2+||V3-X+D4||_F^2++||V4-X+D5||_F^2)
%%

[H,W,Dim] = size(Y);
% Dtemp=reshape(Y,[H*W,Dim])';
% norm_two = lansvd(Dtemp, 1, 'L');
% norm_inf = norm( Dtemp, inf) / beta;
% dual_norm = max(norm_two, norm_inf);
% Y = Y / dual_norm;

maxIter = 400;
mu = mu; % 1e-3
mu_bar = 1e10;
rho = 1.5; % 1.5
%%
% fft2_slice = @(X) fft(fft(X,[],2),[],1);
% ifft2_slice=@(X) ifft(ifft(X,[],2),[],1);


%%

% Dh=psf2otf([1,-1],[H,W,Dim]);
% Dw=psf2otf([1;-1],[H,W,Dim]);
% denh = (tau*abs(Dh).^2+mu);
% denw = (tau*abs(Dw).^2+mu);
temp=[-ones(H,1),ones(H,1)];
delta_H=spdiags(temp,[0,-1],H,H-1);
HTH=(delta_H*delta_H');
temp=[-ones(W,1),ones(W,1)];
delta_W=spdiags(temp,[0,-1],W,W-1);
WTW=(delta_W*delta_W');

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------


X = zeros(H,W,Dim);
S=X;
%%%%aux variables
V1=X;
V2=X;
V3=X;
V4=V3;
% Lagrange Multipliers
D1=X;
D2=X;
D3=X;
D4=X;
D5=X;
A=zeros(Dim,max(truncate_rank,1));
B=zeros(H*W,max(truncate_rank,1));
sigma2=0;
area0=0;
%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
iter = 1;
res = inf*ones(5,1);

while (iter <= maxIter) && (sum(abs(res)) > tol2)
   %% update X
    X = (Y-S+D1+D2+D3+D4+V1+V2+V3)/4;  
  
   %% update V1
    X1=Tensor_unfold(X-D2,1);
    M1=mu*X1/(tau*HTH+mu*eye(H));
    V1=fold_k(M1,1,[H,W,Dim]);
    %V1=real(ifft2_slice(fft2_slice(mu*(X-D2))./denh));
   %% update V2
    X2=Tensor_unfold(X-D3,2);
    M2=mu*X2/(tau*WTW+mu*eye(W));
    V2=fold_k(M2,2,[H,W,Dim]);
%     V2=real(ifft2_slice(fft2_slice(mu*(X-D3))./denw));
   %% update V3
    temp = 0.5*(X+V4-D4+D5);
    temp3=Tensor_unfold(temp,3)';
    [Us,sigma,Vs] = svd(temp3,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>(alphia/(2*mu))));
    if svp >= 1
        sigma = sigma(1:svp)-(alphia/(2*mu));
    else
        svp = 1;
        sigma = 0;
    end
    V3_3 = Us(:,1:svp)*diag(sigma)*Vs(:,1:svp)';  
    V3=fold_k(V3_3',3,[H,W,Dim]);
   %% update V4
   temp3=fold_k(alphia*B*A'/mu,3,[H,W,Dim]);
   V4=V3-D5+temp3;
   %% update A ,B
   temp3=Tensor_unfold(V4,3)';
 
   if truncate_rank>=1
%        [A,sigma2,B]=lansvd(temp3,truncate_rank,'L');
%    else
       [A,sigma2,B]=svds(temp3,truncate_rank);
   end
   if sigma2(1,1)==0 || truncate_rank==0
       A=zeros(Dim,max(truncate_rank,1));
       B=zeros(H*W,max(truncate_rank,1));
   end

    %% update S
    S = solve_l1l1l2(Y-X+D1,beta/mu);
   % figure(5),imshow(sum(abs(X),3),[]);
    %% update D1,D2,D3,D4,D5
     D1=D1+(Y-X-S);
     D2=D2-X+V1;
     D3=D3-X+V2;
     D4=D4-X+V3;
     D5=D5-V3+V4;
     %%
     if mod(iter,10) == 1      
        t0=Y-X-S;
        t0=t0(:);
        res(1)=sqrt(t0'*t0);
        t1=X-V1;
        t1=t1(:);
        res(2)=sqrt(t1'*t1);
        t2=X-V2;
        t2=t2(:);
        res(3)=sqrt(t2'*t2);
        t3=X-V3;
        t3=t3(:);
        res(4)=sqrt(t3'*t3);
        t4=V3-V4;
        t4=t4(:);
        res(5)=sqrt(t4'*t4);

        f_show=sqrt(sum(S.^2,3));
        r_max = max(f_show(:));
        taus = linspace(0, r_max, 5000);
        for index2 = 1:length(taus)
          tau1 = taus(index2);
          anomaly_map_rx = (f_show(:)> tau1)';
          PF0(index2) =sum(anomaly_map_rx & normal_map)/sum(normal_map);
          PD0(index2) =sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
        end
        id=(iter-1)/10+1;
         area0(id)=sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
         RES(id)=res(1);
         % disp(['iter =',num2str(iter),'- res(1) =',num2str(res(1)), ',res(2) =',num2str(res(2)),',res(3) =', num2str(res(3)),',res(4) =', num2str(res(4)),',res(5) =', num2str(res(5)),',AUC=',num2str(area0(id))]);
%   disp(['iter =',num2str(iter),'- res(1) =',num2str(res(1)), ',res(2) =',num2str(res(2)),',res(3) =', num2str(res(3)),',res(4) =', num2str(res(4)),',res(5) =', num2str(res(5))]);
    end  
    iter = iter + 1;    
    mu = min(mu*rho, mu_bar); 
end
% f_show=sqrt(sum(S.^2,3));
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau1 = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau1)';
%   PF0(index2) =sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD0(index2) =sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area0=sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
 area0=area0(end);
end

function [E] = solve_l1l1l2(X,lambda)
[H,W,D] = size(X);
nm=sqrt(sum(X.^2,3));
nms=max(nm-ones(H,W)*lambda,0);
sw=repmat(nms./nm,[1,1,D]);
E=sw.*X;
end

