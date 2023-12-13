function Y = log(X)
% LOG    Natural logarithm.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if isreal(X)
    
    % X is a real quaternion, and we compute the logarithm of an isomorphic
    % complex number using the standard Matlab log function, then construct
    % a quaternion with the same axis as the original quaternion.
    
    Y = isoquaternion(log(isocomplex(X)), X);
else
    
    % X is a complex quaternion, and therefore we cannot use the method
    % above for real quaternions, because it is not possible to construct
    % an isomorphic complex number. The algorithm used is documented in the
    % appendix.

    Y = quaternion(log(modsquared(X))./2, axis(X) .* angle(X));
end;

% Appendix.
%
% The calculation of log(X) is not difficult to derive, as follows.
%
% First note that exp(Y) = X (definition of logarithm). Then write
% X in polar form as r .* exp(mu .* theta). Then we have:
%
% log(X) = Y = log(r) + log(exp(mu .* theta)) = log(r) + mu .* theta.
% 
% Rather than calculate r = abs(X), we use modsquared because this
% avoids calculating a square root. We compensate by halving log(r).
% mu and theta are obtained using the axis and angle functions. Note
% that we don't need to add the two results, since the first is a
% real or complex value, and the second is a pure quaternion (real
% or complex). Therefore we can compose them using the quaternion
% constructor function.

% $Id: log.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

