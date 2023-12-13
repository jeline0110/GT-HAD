function tf = isreal(A)
% ISREAL True for real (quaternion) array.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function returns true if all the components of A are real, that is,
% A is a quaternion with real coefficients (a real quaternion).

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if ispure_internal(A)
    tf = isreal(exe(A)) & isreal(wye(A)) & isreal(zed(A));
else
    tf = isreal(ess(A)) & isreal(vee(A));
end

% $Id: isreal.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

