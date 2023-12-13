function tf = ishermitian(A, tol)
% ISHERMITIAN  True if the given matrix is Hermitian to within the tolerance
% given (optionally) by the second parameter.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 1
    tol = 4 .* eps; % The tolerance was not specified, supply a default.
end

[r, c] = size(A);

if r ~= c
    error('Cannot test whether a non-square matrix is Hermitian.');
end

tf = ~any(any(abs(vector(A + A.')) ./ max(max(abs(A))) > tol));

% $Id: ishermitian.m,v 1.2 2009/02/08 18:35:21 sangwine Exp $

