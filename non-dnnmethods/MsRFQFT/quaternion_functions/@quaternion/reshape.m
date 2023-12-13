function a = reshape(q, varargin)
% RESHAPE Change size.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargoutchk(0, 1, nargout))

a = overload(mfilename, q, varargin{:});

% $Id: reshape.m,v 1.5 2009/02/08 18:35:21 sangwine Exp $

