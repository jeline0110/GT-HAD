function tf = isinf(A)
% ISINF  True for infinite elements.  
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if ispure_internal(A)
    tf = isinf(exe(A)) | isinf(wye(A)) | isinf(zed(A));
else
    tf = isinf(ess(A)) | isinf(vee(A));
end

% $Id: isinf.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

