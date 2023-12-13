function p = convert(q, t)
% CONVERT Convert elements of a quaternion to another type.
% See also cast, which overloads the standard Matlab function.
%
% This function converts a quaternion array into a quaternion array with
% components of a different data type (e.g. int8, double, single).

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~ischar(t)
    error('Second parameter must be a string.')
end

f = str2func(t); % Construct a function handle from t, which must denote
                 % a function on the current Matlab path, so if it does
                 % not, an error will occur here.
if ispure_internal(q)
    p = class(compose(f(exe(q)), f(wye(q)), f(zed(q))), 'quaternion');
else
    p = class(compose(f(ess(q)), f(exe(q)), ...
                      f(wye(q)), f(zed(q))), 'quaternion');
end

% $Id: convert.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

