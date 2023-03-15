function B = squeeze_my2(A)
[n3,n2,n1] = size(A);
B = zeros(n1,n2,n3);
for i = 1 : n3
   slice = A(i,:,:);
   B(:,:,i) = reshape(slice,n1,n2);    
end
