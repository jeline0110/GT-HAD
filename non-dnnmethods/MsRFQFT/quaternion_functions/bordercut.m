function [sM] = bordercut(sM,BorderCutValue)

[m n] = size(sM);

delta=ceil(BorderCutValue*m);

sM(:,1:delta) = 0;
sM(1:delta,:) = 0;
sM(m-delta+1:m,:) = 0;
sM(:,n-delta+1:n) = 0;
