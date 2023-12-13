function tf = ispure(q)
% ISPURE   Tests whether a quaternion is pure.
% tf = ispure(q) returns 1 if q has no scalar part, 0 otherwise. Note that
% if q has a scalar part which is zero, ispure(q) returns 1. Also,
% ispure(q) returns 1 if q is an empty quaternion, since it has no scalar
% part.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

% Implementation note. This function uses ispure_internal, so that the
% knowledge of how quaternions are represented is hidden in the private
% functions.

tf = ispure_internal(q);

% $Id: ispure.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

