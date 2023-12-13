function tf = isunitary(A, tol)
% ISUNITARY  True if the given matrix is unitary to within the tolerance
% given (optionally) by the second parameter.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 1
    tol = 16 .* eps; % The tolerance was not specified, supply a default.
end

[r, c] = size(A);

if r ~= c
    error('A non-square matrix cannot be unitary.');
end

% The method used is to subtract a quaternion identity matrix from A * A'.
% The result should be almost zero.  To compare it against the tolerance,
% we add the moduli of the four components.  This is guaranteed to give a
% real result, even when A is a complexified quaternion matrix.

D = A * A' - quaternion(eye(r));

tf = all(all(abs(s(D)) + abs(x(D)) + abs(y(D)) + abs(z(D)) < tol));

% $Id: isunitary.m,v 1.2 2009/02/08 18:35:21 sangwine Exp $

