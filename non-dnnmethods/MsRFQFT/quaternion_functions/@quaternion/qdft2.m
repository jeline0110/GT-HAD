function Y = qdft2(X, A, L)
% QDFT2 Discrete quaternion 2D Fourier transform.
%
% This function computes the two-dimensional discrete quaternion Fourier
% transform of X, which may be a real or complex quaternion matrix.
% A is the transform axis and it may be a real or complex pure quaternion.
% It need not be a unit pure quaternion. L may take the values 'L' or 'R'
% according to whether the hypercomplex exponential is to be multiplied
% on the left or right of X.
%
% This function uses direct evaluation using a matrix product, and it is
% intended mainly for verifying results against fast transform
% implementations such as qfft2.m. See also: iqdft2.m.

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

% Compute the transform. This is done by row/column separation, that is we
% compute the QDFT of the rows, then the QDFT of the columns. This is
% faster than a direct implementation, and easier, because the direct
% implementation would require a block matrix for the exponentials, which
% Matlab cannot support.

Y = qdft(qdft(X, A, L).', A, L).';

% $Id: qdft2.m,v 1.8 2009/02/08 18:35:21 sangwine Exp $

