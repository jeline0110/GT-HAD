function Y = expm(X)
% Matrix exponential.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

Y = overloadm(mfilename, X);

% TODO Implement a more accurate dedicated algorithm for this function.

% $Id: expm.m,v 1.4 2010/01/15 18:21:04 sangwine Exp $
