function Y = qdft(X, A, L)
% QDFT Discrete quaternion Fourier transform.
%
% This function computes the one-dimensional discrete quaternion Fourier
% transform of X, which may be a real or complex quaternion array. If X
% is a vector the result is a vector of the same orientation. If X is a
% matrix, the transform is applied to columns of X. (Cf the Matlab dft
% function.)
%
% A is the transform axis and it may be a real or complex pure
% quaternion. It need not be a unit pure quaternion. L may take the values
% 'L' or 'R' according to whether the hypercomplex exponential is to be
% multiplied on the left or right of X. There are no default values.
%
% This function uses direct evaluation using a matrix product, and it is
% intended mainly for verifying results against fast transform
% implementations such as qfft.m. See also: iqdft.m.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(3, 3, nargin)), error(nargoutchk(0, 1, nargout))

if ~isscalar(A)
    error('The transform axis cannot be a matrix or vector.');
end

if ~isa(A, 'quaternion') || ~ispure_internal(A)
    error('The transform axis must be a pure quaternion.')
end

if L ~= 'L' && L ~= 'R'
    error('L must have the value ''L'' or ''R''.');
end

A = unit(A); % Ensure that A is a unit (pure) quaternion.

% If X is a vector we need to make sure that we work with a column vector
% internally, and return a result in the same orientation as X.

if isvector(X)
    r = length(X);
    row = (all(size(X) == [1, r]));
    if row
        X = X.';
    end
else
    r = size(X, 1); % X is a matrix, r is the number of columns.
end

E = exp(-A .* 2 .* pi .* ((0:r-1)' *(0:r-1)) ./r);

if L == 'L'
    Y = E * X; % Multiply the exponential matrix on the left.
elseif L == 'R'
    Y = (X.' * E.').'; % To multiply the exponential matrix on the right
                       % we transpose both and transpose the result.
else
    error('L has incorrect value');
end

if isvector(X) && row
   Y = Y.'; %% Make Y the same orientation as the original X. 
end

% $Id: qdft.m,v 1.9 2009/02/08 18:35:21 sangwine Exp $

