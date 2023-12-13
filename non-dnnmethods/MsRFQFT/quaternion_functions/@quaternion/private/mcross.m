function c = mcross(a, b)
% Matrix cross (vector) product of two pure quaternions. Like cross but matrix
% not elementwise.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isa(a, 'quaternion') || ~isa(b, 'quaternion')
    error('Matrix cross product is not defined for a quaternion and a non-quaternion.')
end

if ~ispure_internal(a) || ~ispure_internal(b)
    error('Mcross product is defined only for pure quaternions.')
end

c = class(compose(wye(a) * zed(b) - zed(a) * wye(b),   ...
                  zed(a) * exe(b) - exe(a) * zed(b),   ...
                  exe(a) * wye(b) - wye(a) * exe(b)), 'quaternion');

% $Id: mcross.m,v 1.4 2009/02/08 17:04:59 sangwine Exp $
