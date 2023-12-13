function d = uint32(q)
% UINT32 Convert to unsigned 32-bit integer (obsolete).
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(['Conversion to uint32 from quaternion is not possible. ',...
       'Try cast(q, ''uint32'')'])

% Note: this function was replaced from version 0.9 with the convert
% function, because it is incorrect to provide a conversion function
% that returns a quaternion result.

% $Id: uint32.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

