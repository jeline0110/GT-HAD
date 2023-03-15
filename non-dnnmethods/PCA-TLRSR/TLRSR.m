function [ Z,E,Z_rank,err ] = TLRSR(X,A,max_iter,lambda,Debug)

[n1,n2,n3]=size(X);
[~,n4,~]=size(A);

%% Z=J=Y1 n4 n2 n3
Z=zeros(n4,n2,n3);
J=Z;
Y1=Z;

%% E=Y2 n1 n2 n3
E = zeros(n1,n2,n3);
Y2=E;


beta = 1e-4;
max_beta = 1e+8;
tol = 1e-8;
rho = 1.1;
iter = 0;

a1=zeros(n1,n2,n3);
a1(:,:,1)=eye(n1,n2);

Ain = t_inverse(A);
AT = tran(A);
while iter < max_iter
    iter = iter+1;
    
    %% update Zk
    Z_pre = Z;
    R1 = J-Y1/beta;
    [Z,Z_nuc,Z_rank] = prox_tnn_w(R1,1/beta);
    
    
    %% update Ek
    E_pre = E;
    R2=X-tprod(A,J)+Y2/beta;
    E = solve_l1l1l2( R2, lambda/beta );
    
    
    %% update Jk
    J_pre=J;
    Q1=Z+Y1/beta;
    Q2=X-E+Y2/beta;
    J=tprod(Ain, Q1+tprod(AT,Q2));

    
    %% check convergence
    leq1 = Z-J;
    leq2 = X-tprod(A,J)-E;
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));
    
    difJ = max(abs(J(:)-J_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difZ = max(abs(Z(:)-Z_pre(:)));
    err(iter) = max([leqm1,leqm2,difJ,difZ,difE]);
    if (Debug && (iter==1 || mod(iter,20)==0))
        sparsity=length(find(E~=0));
        fprintf('iter = %d, obj = %.3f, err = %.8f, beta=%.2f, rankL = %d, sparsity=%d\n'...
            , iter,Z_nuc+lambda*norm(E(:),1),err,beta,Z_rank,sparsity);
    end
    if err < tol
        break;
        iter
    end
    
    %% update Lagrange multiplier and  penalty parameter beta
    Y1 = Y1 + beta*leq1;
    Y2 = Y2 + beta*leq2;
    beta = min(beta*rho,max_beta);
end
