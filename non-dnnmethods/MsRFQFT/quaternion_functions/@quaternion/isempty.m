function tf = isempty(q)
% ISEMPTY True for empty matrix.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

% It is sufficient to check the x component, because if x is empty so must
% be the y and z components. We must not check the scalar component,
% because this is empty for a non-empty pure quaternion.
     
tf = isempty(q.x); 

% $Id: isempty.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

