function [ ss,sss ] = t_full_svd( Y )
[n1,n2,n3] = size(Y);
% n12 = min(n1,n2);
Y = fft(Y,[],3);
% U = zeros(n1,n12,n3);
% V = zeros(n2,n12,n3);
% S = zeros(n12,n12,n3);
ss=[];
sss=0;
for i = 1 : n3
    [~,s,~] = svd(Y(:,:,i));
    s=diag(s);
    ss = [ss;s];
    sss=sss+s;
end
end

