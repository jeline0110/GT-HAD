function e = tubebasis(n3,k)
% tube basis
% output a tensor of size 1*1*n3 with its (1,1,k)-th entry equaling to 1 and the rest equaling to 0. 

e = zeros(1,1,n3);
e(1,1,k) = 1;