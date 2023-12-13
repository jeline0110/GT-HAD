function Y = iqdft2(X, A, L)
% IQDFT2 Inverse discrete quaternion 2D Fourier transform.
%
% This function computes the inverse discrete quaternion Fourier transform
% of X. See the function qdft2.m for details. Because this is an inverse
% transform it divides rows and columns of the result by their lengths.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(3, 3, nargin)), error(nargoutchk(0, 1, nargout))

% We omit any check on the A and L parameters here, because they will be
% checked by the qdft code called by iqdft.

% Compute the transform. This is done by row/column separation, that is we
% compute the QDFT of the rows, then the QDFT of the columns. This is
% faster than a direct implementation, and easier, because the direct
% implementation would require a block matrix for the exponentials, which
% Matlab cannot support.

Y = iqdft(iqdft(X, A, L).', A, L).';

% $Id: iqdft2.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

