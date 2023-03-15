function [ inv_a ] = t_inverse(A)

[~,n4,n3]=size(A);

A = fft(A,[],3);
inv_a = zeros(n4,n4,n3);
for i = 1 : n3
   inv_a(:,:,i) =  (A(:,:,i)'*A(:,:,i) + eye(n4))\eye(n4);
end
% inv_a = (Abar'*Abar + eye(m*d2))\eye(m*d2);
inv_a = ifft(inv_a,[],3);

%U = zeros(n1,n12,n3);
% B = zeros(n4,n4,n3);
% %S = zeros(n12,n12,n3);
% %trank = 0;
% ss=ones(n4,1);
% n=min(n1,n4);
% for i = 1 : n3
% %    [U(:,:,i),s,V(:,:,i)] = svd(A(:,:,i));
%     [~,s,V] = svd(A(:,:,i));
%     s = diag(s);
%     s=1./(s+1);
%     ss(1:n)=s;
%     B(:,:,i)=V*diag(ss)*V';
% end
% 
% B = ifft(B,[],3);
end

