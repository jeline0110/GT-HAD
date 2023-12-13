function n = det(X, d)
% DET    Determinant.
% (Quaternion overloading of standard Matlab function, with differences.)
%
% The second parameter may be:
%
% 'Moore'     The determinant is the product of the eigenvalues of X.
% 'DieudonnéŽ' or
% 'Dieudonne' The determinant is the product of the singular values of X.
% 'Study'     The determinant is the determinant of the adjoint of X. If X
%             is a real quaternion matrix, the complex adjoint is used. If
%             X is a complexified quaternion matrix, the real adjoint is
%             used.
%
% The Moore determinant is the only one which can be negative or complex.
% but it cannot be calculated for a non-Hermitian matrix since there is
% currently no way to calculate the eigenvalues of a non-Hermitian matrix.
% The Study determinant is the square of the DieudonnéŽ determinant and is
% much faster to calculate, but it may not be accurately real, whereas the
% DieudonnŽé determinant is real by definition.
%
% For the moment, there is no default value for the second parameter. This
% may be changed if a way is found to calculate the Moore determinant for
% non-Hermitian matrices (Moore will be the default).

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% References:
%
% Helmer Aslaksen, 'Quaternionic Determinants',
% The Mathematical Intelligencer, 18 (3), 1996, 56-65.
% [Reprinted in 'Mathematical Conversation: Selections from
%  The Mathematical Intelligencer', Robin Williams and Jeremy Gray (eds.),
%  142-156, Springer-Verlag, 2001.]
%
% F. Z. Zhang, Quaternions and Matrices of Quaternions,
% Linear Algebra and its Applications, 251, January 1997, 21-57. [see p47]

% Note:  Aslaksen defines the complex adjoint matrix differently to Zhang
% but the two definitions are in fact equivalent,  since Aslaksen defines
% the Cayley_Dickson form with j on the left, whereas Zhang puts j on the
% right.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

[r,c] = size(X);

if r ~= c
    error('Matrix must be square.');
end

% For the moment, there is no default, and the error check on nargin
% above requires 2 arguments. If the restriction to Hermitian matrices
% can be removed, this code can be reinstated, and the error check
% alterered to 1, 2.
%
% if nargin == 1
%     d = 'Moore'; % Set the default value if none was supplied.
% end

switch d
case {'DieudonnéŽ', 'Dieudonne'}
    
    % The DieudonnŽe determinant is defined by diagonalising the matrix.
    % Since the determinant of a product is the product of the determinants
    % and the determinant of a unitary matrix is 1, we can calculate the
    % DieudonnŽe determinant using the SVD (it is the product of the
    % singular values).

    n = prod(svd(X));
case 'Study'

    % The Study determinant is defined in terms of the adjoint, but we
    % cannot compute a complex adjoint for a complexified quaternion
    % matrix - we must use a real adjoint. Of course, Study did not
    % consider this, so what is coded below is an extension of his idea.
    % The Study determinant is real for a quaternion matrix, and complex
    % for a complexified quaternion matrix. The Study determinant is the
    % square of the Dieudonné determinant.

    if any(any(imag(X) ~= 0))
        n = det(adjoint(X, 'real'));
    else
        n = det(adjoint(X));
    end
case 'Moore'

    % The Moore determinant is the product of the eigenvalues, but the
    % matrix must be Hermitian (implementation restriction on the eig
    % function which may be removed in the future).

    if ishermitian(X)
        n = prod(eig(X));
    else
        error('Cannot compute Moore determinant of a non-Hermitian matrix.')
    end
otherwise
    error(['Unrecognized second parameter, ', d, ' determinant unknown.']);
end


% $Id: det.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

