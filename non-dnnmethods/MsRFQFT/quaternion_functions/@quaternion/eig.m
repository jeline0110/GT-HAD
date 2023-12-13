function [V, D] = eig(A)
% EIG    Eigenvalues and eigenvectors.
% (Quaternion overloading of standard Matlab function, with limitations.)
%
% Acceptable calling sequences are: [V,D] = EIG(X) and V = EIG(X).
% The results are as for the standard Matlab EIG function (q.v.).

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 2, nargout))

[r, c] = size(A);

if r ~= c
    error('Matrix must be square');
end

if r == 1
    
    % The argument is a single quaternion. This case could be handled by
    % using the standard Matlab eig() function on the complex adjoint of
    % A, but there are problems if A is a complexified quaternion, since
    % we cannot make a complex value with complex parts.
    %
    % For this reason we output an error message and leave it to the user
    % to use the appropriate adjoint.
    
    A = inputname(1); if A == ''; A = 'A'; end
    
    disp('The eig function is not implemented for single quaternions.');
    disp(sprintf('Try using eig(adjoint(%s, ''real'')) or', A));
    disp(sprintf('          eig(adjoint(%s, ''complex'')',  A));
    error('Implementation restriction - see advice above');
end

% The method used depends on whether A is Hermitian or otherwise.

if ishermitian(A)
    
    % For a Hermitian matrix the eigenvalues and eigenvectors are real if A is a
    % real quaternion matrix and complex if A is a complexified quaternion matrix.
    % The first step is to tridiagonalize A using Householder transformations to
    % obtain P and B where B is a real or complex tridiagonal symmetric matrix,
    % and P is the product of the Householder matrices used to compute B
    % such that P'*B*P = A.

    [P, B] = tridiagonalize(A);
else
    
    % For a general (i.e. non Hermitian) matrix, we don't currently have a method to
    % convert A to a real matrix. The only feasible approach is to use an adjoint
    % matrix, but we leave this to the user.
    
    disp('The eig function is not implemented for non-Hermitian quaternion matrices.');
    disp('Eigenvalues and eigenvectors can be computed by using the standard Matlab');
    disp('eig function on an adjoint matrix by using the function adjoint() (q.v.).');
    error('Implementation restriction - see advice above');

end

% We now have: P' * B * P = A.

% The second step is to compute the eigenvectors and eigenvalues of B using the standard
% Matlab routine, for a real or complex matrix (which happens to be tridiagonal).

if nargout == 0
    eig(B)
elseif nargout == 1
    V = eig(B);
else    
    [V, D] = eig(B);
    V = P' * V; % Combine P (from the tridiagonalization) with V.
end



% $Id: eig.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

