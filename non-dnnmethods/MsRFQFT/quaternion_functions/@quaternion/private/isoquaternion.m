function q = isoquaternion(z, a)
% ISOQUATERNION  Construct a quaternion from a complex number, preserving
%                the modulus and argument, and using the axis of the second
%                argument as the axis of the result.
%
% Copyright © 2006, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isa(a, 'quaternion')
    error('Second argument to private function isocomplex must be a quaternion.');
end;

if isa(z,'quaternion')
    error('First argument to private function isocomplex must not be a quaternion.');
end;

if ~isnumeric(z)
    error('First argument to private function isocomplex must be numeric.');
end

re = imag(z) == 0; % This gives an array of logicals indicating which
er = ~re;          % elements of z lack an imaginary part. We use this to
                   % select which of the two following methods is used to
                   % construct the equivalent quaternion.
    
% 1. If the imaginary part of an element of z is zero, so must be the
%    vector part of the result. In this case we avoid calling the axis
%    function, because axis(a) may be undefined, and we don't need it
%    in any case.
% 2. If the imaginary part of an element of z is non-zero, we use the more
%    general code, which calculates the axis of the argument a (at the
%    corresponding array index), and uses this axis in constructing the
%    quaternion result.

q = zerosq(size(z), class(z)); % Preconstruct the result array.

% This is a private class function. We have to use subsref, subsasgn and
% substruct to handle the indexing here, because normal indexing notation
% does not work in a class function.
% See the file 'implementation notes.txt', item 8, for more details.

% It is possible for the logical arrays re or er to be all zero, in which
% case we must not try to index into the complex or quaternion array, as
% the subsasgn will select no elements.
% TODO Consider how to handle arrays of more than 2 dimensions
% (any(any(any...???

if any(any(re))
    q = subsasgn(q, substruct('()', {re}), ...
        quaternion(real(z(re)))); % A zero vector part will be supplied.
end;
if any(any(er))
    q = subsasgn(q, substruct('()', {er}), ...
        quaternion(real(z(er)), ...
                   imag(z(er)) .* axis(subsref(a, substruct('()', {er})))));
end

% Note: the complex argument z may be in any of the four quadrants of the
% plane, and so may the quaternion result. This means that if the axis is
% extracted from the quaternion result, it may point in the opposite
% direction to the axis of the second argument, a.

% $Id: isoquaternion.m,v 1.6 2010/03/23 14:16:13 sangwine Exp $
