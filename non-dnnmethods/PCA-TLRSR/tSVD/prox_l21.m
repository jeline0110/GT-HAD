function [ Q, sparsity, supp_set,l21 ] = prox_l21( Q, rho )
[~,n2,~]=size(Q);
sparsity=0;
supp_set=[];
l21=0;
for i=1:n2
    QF=sum(sum(Q(:,i,:).*Q(:,i,:)))^0.5;
    if QF>rho
        ratio=1-rho/QF;
        Q(:,i,:)=ratio*Q(:,i,:);
        sparsity=sparsity+1;
        supp_set=[supp_set,i];
        l21=l21+ratio*QF;
    else
        Q(:,i,:)=0.0*Q(:,i,:);
    end
end
end

