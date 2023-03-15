function [ Y,trank,U,V,S ] = prox_low_rank(Y,tho)
[n1,n2,n3] = size(Y);
n12 = min(n1,n2);
Y = fft(Y,[],3);
U = zeros(n1,n12,n3);
V = zeros(n2,n12,n3);
S = zeros(n12,n12,n3);

trank = 0;
for i = 1 : n3
    
    [U(:,:,i),s,V(:,:,i)] = svd(Y(:,:,i),'econ');
    
    r=rank(Y(:,:,i));
    s = diag(s);
    ss=s(1:r-2)./s(2:r-1);
    
    tranki=find(ss>tho);
    if isempty(tranki)
        tranki=r;
    else if length(tranki)>1
           tranki=tranki(1);
        end
    end
            
            
    [p,idx]=max(ss);
    ratio=(r-2)*p/(sum(ss)-p);
    if ratio>50
        tranki=min(tranki,idx);
    end
%     idx=find(s>=tho*s(1));
%     r=rank(Y(:,:,i));
%     tranki=min(length(idx),r);
    
    s=s(1:tranki);  
    S(1:tranki,1:tranki,i) = diag(s);
    U(:,1:tranki,i)=U(:,1:tranki,i);
    V(:,1:tranki,i)=V(:,1:tranki,i);
    trank = max(tranki,trank);
end
U = U(:,1:trank,:);
V = V(:,1:trank,:);
S = S(1:trank,1:trank,:);

U = ifft(U,[],3);
S = ifft(S,[],3);
V = ifft(V,[],3);
Y = tprod( tprod(U,S), tran(V));
end

