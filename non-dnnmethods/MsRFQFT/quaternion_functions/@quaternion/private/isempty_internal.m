function x = isempty_internal(q)
% Determines whether q is empty. This function has to know the
% representation of a quaternion, whereas the function isempty()
% does not.

% THE FUNCTION ISEMPTY has been modified to test the X component. This does
% not require the internal structure to be known, and therefore this
% function is no longer required. For the moment it will be marked as
% redundant and removed later.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error('isempty_internal is redundant and should not be called')

% x = isempty(q.w) & isempty(q.x) & isempty(q.y) & isempty(q.z);

% Implementation note. Since a quaternion cannot have a mix of empty and
% non-empty components (apart from the case of a pure quaternion, where the
% scalar part is empty), it is not necessary to test all three of the x, y,
% z components. So, maybe it would be OK here to code this as:
%
%       x = isempty(q.x);

% TODO It is also possible this could be coded as length(q) == 0, using the
% size function, but maybe this would be inefficient compared to the above
% code.

% $Id: isempty_internal.m,v 1.5 2009/02/08 17:04:59 sangwine Exp $
