function n = ndims(x)
% NDIMS   Number of array dimensions.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

n = ndims(exe(x));

% An alternative, previously used is:
%
% n = length(size(q));
%
% but using the builtin function on the x-component is simpler, c.f. the
% end function, where the same approach is used.

% $Id: ndims.m,v 1.5 2009/02/08 18:35:21 sangwine Exp $

