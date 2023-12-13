function c = vector_product(a, b)
% VECTOR (cross) PRODUCT of two pure quaternions.

% Copyright © 2005-8 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isa(a, 'quaternion') || ~isa(b, 'quaternion')
    error('Vector/cross product is not defined for a quaternion and a non-quaternion.')
end

if ~ispure_internal(a) || ~ispure_internal(b)
    error('Vector/cross product is defined only for pure quaternions.')
end

c = class(compose(a.y .* b.z - a.z .* b.y, ...
                  a.z .* b.x - a.x .* b.z, ...
                  a.x .* b.y - a.y .* b.x), 'quaternion');

% $Id: vector_product.m,v 1.5 2009/10/31 17:53:46 sangwine Exp $

