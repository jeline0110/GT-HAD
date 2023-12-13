function X = sqrtm(A)
% Matrix square root.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

X = overloadm(mfilename, A);

% TODO Implement a more accurate dedicated algorithm for this function.

% $Id: sqrtm.m,v 1.4 2010/01/15 18:21:04 sangwine Exp $
