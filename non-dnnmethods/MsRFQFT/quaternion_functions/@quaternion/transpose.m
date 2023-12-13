function t = transpose(a)
% .'  Transpose.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

t = overload(mfilename, a);

% $Id: transpose.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

