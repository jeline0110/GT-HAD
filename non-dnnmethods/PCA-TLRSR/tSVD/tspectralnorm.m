function norm2 = tspectralnorm(C)
% Tensor spectral norm
% C - n1*n2*n3 tensor
% norm2 - tensor spectral norm


C = fft(C,[],3);
norm2 = zeros(size(C,3),1);
for i = 1 : size(C,3)
   norm2(i) = norm(C(:,:,i),2);
end
norm2 = max(norm2);


