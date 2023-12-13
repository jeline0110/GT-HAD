function tf = isnan(A)
% ISNAN  True for Not-a-Number.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if ispure_internal(A)
    tf = isnan(exe(A)) | isnan(wye(A)) | isnan(zed(A));
else
    tf = isnan(ess(A)) | isnan(vee(A));
end

% $Id: isnan.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

