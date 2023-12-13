function b = repmat(a, m, n)
% REPMAT Replicate and tile an array.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 3, nargin)), error(nargoutchk(0, 1, nargout)) 

if nargin == 2
    b = overload(mfilename, a, m);
else
    b = overload(mfilename, a, m, n);
end

% $Id: repmat.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

