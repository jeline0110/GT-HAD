function d = diag(v, k)
% DIAG Diagonal matrices and diagonals of a matrix.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout)) 

if nargin == 1
    d = overload(mfilename, v);
else
    d = overload(mfilename, v, k);
end

% $Id: diag.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

