function [ U,V,S,trank ] = tSVDs( Y )
[n1,n2,n3] = size(Y);
n12 = min(n1,n2);
Y = fft(Y,[],3);
U = zeros(n1,n12,n3);
V = zeros(n2,n12,n3);
S = zeros(n12,n12,n3);
trank = 0;
for i = 1 : n3
    [U(:,:,i),s,V(:,:,i)] = svd(Y(:,:,i),'econ');
    s = diag(s);
%     s = max(s-rho,0);    
    S(:,:,i) = diag(s);
    tranki = rank(Y(:,:,i));
    trank = max(tranki,trank);
end
U = U(:,1:trank,:);
V = V(:,1:trank,:);
S = S(1:trank,1:trank,:);

U = ifft(U,[],3);
S = ifft(S,[],3);
V = ifft(V,[],3);
%  
% objV = sum(S(:));
% 
% X = tprod( tprod(U,S), tran(V));

% compute rank X
% rs = zeros(n3,1);
% for i = 1 : n3
%     s = S(:,:,i);
%     rs(i) = length(find(s~=0));
% end
% rankX = max(rs);


end

