function [r] = multirank(X)

X = fft(X,[],3);
for i = 1 : size(X,3)
   r(i) = rank(X(:,:,i)); 
end

% figure(1) 
% hold on
% for i = 1 : 4
%    s = svd(X(:,:,i))
%    plot(s)
%   
% end