function a = normq(q)
% NORMQ Norm of a quaternion, the sum of the squares of the components.
% The norm is also equal to the product of a quaternion with its conjugate.

% Copyright © 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

a = modsquared(q);

% Implementation note: the private function modsquared has existed from the
% earliest versions of QTFM. Since it computes exactly what is required
% here it is sensible to use it rather than writing new code. In the longer
% term it might make more sense to move the code of modsquared here, and
% replace all calls to modsquared by calls to normq. This is noted in the
% code for modsquared.

% $Id: normq.m,v 1.1 2009/11/05 10:20:05 sangwine Exp $
