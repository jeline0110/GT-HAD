function Y = diff(X, n, dim)
% DIFF   Differences.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 3, nargin)), error(nargoutchk(0, 1, nargout)) 

if nargin == 1
    Y = overload(mfilename, X);
elseif nargin == 2
    Y = overload(mfilename, X, n);
else
    if ~isnumeric(dim)
        error('Dimension argument must be numeric');
    end
    
    if ~isscalar(dim) || ~ismember(dim, 1:ndims(X))
        error(['Dimension argument must be a positive'...
               ' integer scalar within indexing range.']);
    end

    Y = overload(mfilename, X, n, dim);   
end

% Note: In the absence of this file, the Matlab function of the same name is
% called for a quaternion array, but it raises an error. That is why this
% function was written, and why the Matlab code can't simply be used to
% difference quaternion arrays.

% $Id: diff.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

