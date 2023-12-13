function q = qk
% qk is one of the three quaternion operators.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

q = quaternion(0, 0, 1);

% Implementation note:  we could use k or K for this operator, but this
% would be inconsistent with the other two, where the use of i/I or j/J
% is not possible.

% $Id: qk.m,v 1.2 2009/02/08 19:18:16 sangwine Exp $

