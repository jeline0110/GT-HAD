function x = ess(q)
% Extracts the scalar component of a full quaternion. If q is a pure
% quaternion, an error is raised, since the scalar part of a pure
% quaternion should not be extracted.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if isempty(q.w)
    error('The scalar part of a pure quaternion is not defined.');
end

x = q.w;

% $Id: ess.m,v 1.2 2009/02/08 17:04:59 sangwine Exp $
