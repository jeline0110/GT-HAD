function Y = cos(X)
% COS    Cosine.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if isreal(X)
    
    % X is a real quaternion, and we compute the cosine of an isomorphic
    % complex number using the standard Matlab cos function, then
    % construct a quaternion with the same axis as the original quaternion.
    
    Y = isoquaternion(cos(isocomplex(X)), X);
else
    
    % X is a complex quaternion, and therefore we cannot use the method
    % above for real quaternions, because it is not possible to construct
    % an isomorphic complex number. We use instead a fundamental formula
    % for the cosine in terms of the exponential.
    
    Y = (exp(i .* X) + exp(-i .* X)) ./ 2;
end;

% $Id: cos.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

