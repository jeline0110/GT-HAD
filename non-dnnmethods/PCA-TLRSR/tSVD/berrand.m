function C = berrand(sizes,rho)

C = zeros(sizes);
A = rand(sizes);
ind = A<rho/2;
C(ind) = 1;
ind = A>1-rho/2;
C(ind) = -1;
