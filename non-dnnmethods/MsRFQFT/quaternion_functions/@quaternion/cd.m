function [A, B] = cd(q)
% CD: returns two complex numbers which are the Cayley-Dickson components
% of the quaternion.

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(2, 2, nargout))

if ~isreal(q)
    error('Cannot convert a complex quaternion to Cayley-Dickson form.')
end

A = complex(scalar(q), x(q)); % Note 1.
B = complex(     y(q), z(q));

% Notes
%
% 1. scalar() is used, and not s(), in order to avoid an error if q is
%    pure. Instead, scalar() supplies zero(s) for the scalar part.
%
% 2. In mathematical notation:
%
% q = A + B j where A = w + x i, B = y + z i. Thus:
% q = (w + x i) + (y + z i) j = w + x i + y j + z k
%
% 3. Expressed in Matlab/QTFM code A and B are such that:
%
% q = quaternion(real(A), imag(A), real(B), imag(B)).
% $Id: cd.m,v 1.2 2009/02/08 18:35:21 sangwine Exp $

