function B = ipermute(A, order)
% IPERMUTE Inverse permute dimensions of N-D array
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

B = overload(mfilename, A, order);

% $Id: ipermute.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

