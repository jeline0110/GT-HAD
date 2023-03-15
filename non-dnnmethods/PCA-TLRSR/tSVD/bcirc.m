function bC = bcirc(C)

% Use the Corollary 4 in the paper
% Circulant Matrices and Their Application to Vibration Analysism 2014.
% C - n1*n2*n3 tensor
% bC - block circulant matrix
[n1,n2,n3] = size(C);
s = eye(n3,n3);
bC = zeros(n1*n3,n2*n3);
for i = 1 : n3
    S = gallery('circul',s(i,:)')';
    bC = bC + kron(S,C(:,:,i));    
end