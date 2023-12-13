function tf = isfinite(A)
% ISFINITE  True for finite elements.  
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if ispure_internal(A)
    tf = isfinite(exe(A)) & isfinite(wye(A)) & isfinite(zed(A));
else
    tf = isfinite(ess(A)) & isfinite(vee(A));
end

% $Id: isfinite.m,v 1.5 2010/03/05 17:29:40 sangwine Exp $
