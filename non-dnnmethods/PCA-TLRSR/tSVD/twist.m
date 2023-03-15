function B = twist(A)
[n1,n2,n3] = size(A);
B = zeros(n1,n3,n2);
for i = 1 : n3
   slice = A(:,:,i);
   B(:,i,:) = reshape(slice,n1,1,n2);    
end
