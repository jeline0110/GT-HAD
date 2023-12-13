function P = change_basis(Q, B)
% CHANGE_BASIS changes the basis of the quaternion Q, to the basis B.
% Q may be a vector or matrix of quaternions, or a scalar quaternion.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isa(Q, 'quaternion')
    error('Q must be a quaternion, or a vector or matrix of quaternions.')
end

% Verify that B is an orthonormal basis.

if any(size(B) ~= [3, 3])
    error('The basis, B, must be a 3 by 3 matrix.');
end

if max(max(B * B.' - eye(3))) > 10 * eps
    warning('QTFM:inaccuracy', ...
            'The basis matrix is not accurately orthogonal.');
end

% Construct three pure quaternions from B.

V1 = quaternion(B(1,1), B(1,2), B(1,3));
V2 = quaternion(B(2,1), B(2,2), B(2,3));
V3 = quaternion(B(3,1), B(3,2), B(3,3));

% Change the basis of Q. This is done by resolving the vector part of Q
% into the directions of the three basis vectors V1, V2 and V3.

if ispure_internal(Q)
    P = quaternion(      scalar_product(Q, V1),...
                         scalar_product(Q, V2),...
                         scalar_product(Q, V3));
else
    P = quaternion(s(Q), scalar_product(Q, V1),...
                         scalar_product(Q, V2),...
                         scalar_product(Q, V3));
end

% $Id: change_basis.m,v 1.9 2009/02/08 18:35:21 sangwine Exp $

