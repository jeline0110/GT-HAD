function d = scalar_product(a, b)
% SCALAR_PRODUCT  Quaternion scalar product.
% The scalar (dot) product of two quaternions is the sum of the products of
% the (s, x, y, z) components of the two quaternions.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isa(a, 'quaternion') || ~isa(b, 'quaternion')
    error('Scalar product is not defined for a quaternion and a non-quaternion.')
end

% This function is defined for full and pure quaternions, and combinations
% of full and pure, in which case we assume a zero scalar part for the pure
% argument.

if ispure_internal(a) || ispure_internal(b)
    % This covers the case where either or both are pure. We can ignore the
    % scalar part of the other, since it is implicitly multiplied by zero.
    
    d =              a.x .* b.x + a.y .* b.y + a.z .* b.z;
else
    d = a.w .* b.w + a.x .* b.x + a.y .* b.y + a.z .* b.z;
end

% Note: this function was formerly called 'dot', but this name was inconsistent
% with the Matlab function of the same name which computes the inner product of
% two vectors (real or complex) and overloads to quaternions without
% difficulty. This function, by contrast is a pointwise operator, like the
% cross, or vector product.

% This function will work only for arrays of the same size, or where one or
% both are scalars. At the moment this is not checked, since other
% functions will raise an error if the conditions are not met.

% $Id: scalar_product.m,v 1.5 2009/02/08 18:35:21 sangwine Exp $

