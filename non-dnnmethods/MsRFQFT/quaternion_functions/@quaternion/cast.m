function b = cast(q, newclass)
% CAST  Cast quaternion variable to different data type.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~ischar(newclass)
    error('Second parameter must be a string.')
end

b = overload(mfilename, q, newclass);

% $Id: cast.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

