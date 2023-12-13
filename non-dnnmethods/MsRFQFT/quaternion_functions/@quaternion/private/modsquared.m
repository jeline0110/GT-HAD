function n = modsquared(q)
% Modulus squared of a quaternion.

% Copyright © 2005, 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

if isempty(q.w)
    n =          q.x.^2 + q.y.^2 + q.z.^2;
else
    n = q.w.^2 + q.x.^2 + q.y.^2 + q.z.^2;
end

% TODO Consider moving this code into the user-visible function normq and
% removing modsquared from the toolbox (since it is a private function this
% cannot impact any user code). It would of course be necessary to find and
% replace all calls to modsquared by calls to normq.

% $Id: modsquared.m,v 1.5 2009/11/05 10:55:43 sangwine Exp $
