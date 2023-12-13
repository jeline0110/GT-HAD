function [U,S,V] = svd(X, econ)
% SVD    Singular value decomposition.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% References:
%
% S. J. Sangwine and N. Le Bihan,
% Quaternion Singular Value Decomposition based on Bidiagonalization
% to a Real or Complex Matrix using Quaternion Householder Transformations,
% Applied Mathematics and Computation, 182 (1), 1 November 2006, 727-738.
% DOI:10.1016/j.amc.2006.04.032.
%
% S. J. Sangwine and N. Le Bihan,
% Quaternion Singular Value Decomposition based on Bidiagonalization
% to a Real Matrix using Quaternion Householder Transformations,
% arXiv:math.NA/0603251, 10 March 2006. Available at http://www.arxiv.org/.
%
% Note that the title of the article above refers to a real bidiagonal
% matrix, but the algorithm is valid even when X is a complex quaternion
% matrix, in which case the bidiagonal matrix will be complex. The later
% paper in Applied Mathematics and Computation covers both the real and
% complex cases.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 3, nargout))

if nargin == 2
    if isnumeric(econ) && econ ~= 0
        error('Use svd(X,0) for economy size decomposition.');
    end
    if ~isnumeric(econ) && ~strcmp(econ, 'econ')
        error('Use svd(X,''econ'') for economy size decomposition.');
    end
end

if isscalar(X)
    
    % The argument is a single quaternion. This case could be handled by
    % using the standard Matlab svd() function on the complex adjoint of
    % X, but there are problems if X is a complexified quaternion, since
    % we cannot make a complex value with complex parts.
    %
    % For this reason we output an error message and leave it to the user
    % to use the appropriate adjoint.
    
    X = inputname(1); if X == ''; X = 'X'; end
    
    disp('The svd function is not implemented for single quaternions.');
    disp(sprintf('Try using svd(adjoint(%s, ''real'')) or', X));
    disp(sprintf('          svd(adjoint(%s, ''complex'')',  X));
    error('Implementation restriction - see advice above');
end

% X is not a single quaternion, so proceed ...

% The method used is to bidiagonalize X using Householder transformations
% to obtain H, B, and G where B is a real or complex bidiagonal matrix, and
% H and G are the products of the Householder matrices used to compute B.
% Then: H' * B * G' = X. If X is pure we provide a zero scalar part to
% prevent problems in the bidiagonalization.

if ispure(X)
    [H, B, G] = bidiagonalize(zeros(size(X), class(exe(X))) + X);
else
    [H, B, G] = bidiagonalize(X);
end

% The second step is to compute the SVD of B using the standard Matlab
% routine, for a real or complex matrix (which happens to be bidiagonal).

if nargout == 0

    % The econ parameter is passed to the Matlab svd routine, even though
    % it appears to have no effect if there is no output parameter, since
    % only the singular values are output, and they are output as a column
    % vector.

    if nargin == 1
        svd(B)
    else
        svd(B, econ)
    end

elseif nargout == 1

    % The econ parameter is passed to the Matlab svd routine, even though
    % it appears to have no effect if there is only one output parameter,
    % because the output is a column vector of the singular values.

    if nargin == 1; U = svd(B); else U = svd(B, econ); end

else

    % In this case, the econ parameter has been given, but if we were to
    % pass it to the Matlab svd routine, the returned U and V matrices
    % would not be compatible with H and G. Therefore, we don't pass it,
    % but after multiplying the results by H and G, we truncate them to
    % give the economy mode result, so that the result is compatible with
    % that produced by the standard Matlab svd function.

    [US, S, VS] = svd(B); 
    
    U = H' * US;     % Multiply together the intermediate unitary
    V = (VS' * G')'; % matrices to construct U and V.
                     % Notice that US and VS may be complex and that we
                     % must use ' and not .' here. (H and G are
                     % quaternion-valued, and therefore H' means a
                     % quaternion transpose conjugate.)

    if nargin == 2

        % The economy size decomposition has been demanded, so we have
        % to truncate U or V and S accordingly. The description of how
        % the truncation is done can be found in the Matlab help page
        % for the standard svd function.

        [m, n] = size(X);

        % In what follows, we need to use subscripted references, but
        % these do not work inside a class method (which this is). See the
        % file 'implementation notes.txt', item 8.

        if econ == 0
            if m > n % If m <= n, there is nothing to do.
                U = subsref(U, substruct('()', {':', 1:n})); % U = U(:,1:n)
                S = S(1 : n, 1 : n);
            end
        elseif strcmp(econ, 'econ')
            if m == n
                return
            elseif m > n % The Matlab documentation says >= here, but if m
                         % and n are equal all three matrices are square,
                         % so we don't need to do anything.
                U = subsref(U, substruct('()', {':', 1:n})); % U = U(:,1 n)
                S = S(1 : n, 1 : n);
            else % m < n, since we eliminated m == n above.
                S = S(1 : m, 1 : m);
                V = subsref(V, substruct('()', {1:n, 1:m})); % V = V(1:n,1:m)
            end
        else
            error('econ parameter has incorrect value');
        end
    end
end

% $Id: svd.m,v 1.11 2010/03/22 17:36:59 sangwine Exp $

