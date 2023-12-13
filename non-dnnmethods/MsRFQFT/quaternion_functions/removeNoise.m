function [sM] = removeNoise(sM)

[m n] = size(sM);

sM(:,1:1) = 0;
sM(1:1,:) = 0;
sM(m-0:m,:) = 0;
sM(:,n-0:n) = 0;