function B = twist2(A)
%   horizontal slices correspond to channels
[n1,n2,n3] = size(A);
B = zeros(n3,n2,n1);
for i = 1 : n3
   slice = A(:,:,i);
   B(i,:,:) = reshape(slice,1,n2,n1);    
end
