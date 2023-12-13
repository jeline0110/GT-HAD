function u = uminus(a)
% -  Unary minus.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

if ispure_internal(a)
    u = class(compose(         -exe(a), -wye(a), -zed(a)), 'quaternion');
else
    u = class(compose(-ess(a), -exe(a), -wye(a), -zed(a)), 'quaternion');
end

% $Id: uminus.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

