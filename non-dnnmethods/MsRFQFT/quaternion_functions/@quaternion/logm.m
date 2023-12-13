function L = logm(A)
% Matrix logarithm.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

L = overloadm(mfilename, A);

% TODO Implement a more accurate dedicated algorithm for this function.

% $Id: logm.m,v 1.4 2010/01/15 18:21:04 sangwine Exp $
