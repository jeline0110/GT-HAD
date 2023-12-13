function image(Q)
% IMAGE  Display pure quaternion array as image.
% (Quaternion overloading of standard Matlab function.)
%
% This function supports only one parameter, unlike the overloaded Matlab
% function of the same name, and the parameter supplied must be a pure
% quaternion array. It will be displayed as an image in a graphics window.
% For more sophisticated requirements, convert the quaternion array to a
% suitable double array with three planes and use the Matlab function.

% Copyright © 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 0, nargout))

if ~ispure_internal(Q)
    error('Quaternion image array must be pure.')
end

C = class(exe(Q));
if strcmp(C, 'uint8') || strcmp(C, 'uint16') || strcmp(C, 'double')
    image(cat(3, exe(Q), wye(Q), zed(Q)));
else
    error(['Quaternion array has unknown component class: ', C,...
           ' - expects uint8/16 or double'])
end

% $Id: image.m,v 1.1 2009/03/06 16:59:51 sangwine Exp $
