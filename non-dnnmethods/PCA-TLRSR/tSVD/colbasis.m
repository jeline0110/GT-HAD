function e = colbasis(n,n3,i)
% columnbasis
% output a tensor of size n*1*n3 with its (i,1,1)-th entry equaling to 1 and the rest equaling to 0. 

e = zeros(n,1,n3);
e(i,1,1) = 1;