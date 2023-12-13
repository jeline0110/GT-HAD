function a = round(q)
% ROUND Round towards nearest integer.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

a = overload(mfilename, q);

% $Id: round.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

