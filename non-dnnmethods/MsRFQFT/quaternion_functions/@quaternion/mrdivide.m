function r = mrdivide(a, b)
% /   Slash or right matrix divide.
% (Quaternion overloading of standard Matlab function.)
%
% This function is implemented only for the case in which the
% second parameter is a scalar. In this case, the result is as
% given by the ./ function.   The reason for implementing this
% function is that some Matlab code uses / when ./ should have
% been used.    See, for example, the function cov.m in Matlab
% versions up to (at least) 7.0.4.365 (R14) Service Pack 2.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isscalar(b)
    error('quaternion/mrdivide is implemented only for division by a scalar.');
end

r = a ./ b;

% $Id: mrdivide.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

