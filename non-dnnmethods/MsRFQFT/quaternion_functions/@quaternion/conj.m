function c = conj(a, F)
% CONJ   Quaternion conjugate.
% (Quaternion overloading of Matlab standard function.)
%
% Implements three different conjugates:
%
% conj(X) or
% conj(X, 'hamilton') returns the quaternion conjugate.
% conj(X, 'complex')  returns the complex conjugate.
% conj(X, 'total')    returns the 'total' conjugate equivalent to
%                     conj(conj(X, 'complex'), 'hamilton')

% Copyright © 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 1
    F = 'hamilton'; % Supply the default parameter value.
end

if ~strcmp(F, 'hamilton') && ~strcmp(F, 'complex') && ~strcmp(F, 'total')
    error('Second parameter value must be ''hamilton'', ''complex'' or ''total''.')
end

switch F
    case 'hamilton'
        c = a; c.x = -c.x; c.y = -c.y; c.z = -c.z;
    case 'complex'
        c = overload(mfilename, a); % Conjugate the components of a.
    case 'total'
        c = conj(overload(mfilename, a));
    otherwise
        error('Bad value for second parameter.');
end

% $Id: conj.m,v 1.6 2009/02/08 18:35:21 sangwine Exp $

