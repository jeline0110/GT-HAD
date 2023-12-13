function m = mean(X, dim)
% MEAN   Average or mean value.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout)) 

if nargin == 1
    m = overload(mfilename, X);
else
    if ~isnumeric(dim)
        error('Dimension argument must be numeric');
    end
    
    if ~isscalar(dim) || ~ismember(dim, 1:ndims(X))
        error(['Dimension argument must be a positive'...
               ' integer scalar within indexing range.']);
    end

    m = overload(mfilename, X, dim);
end

% $Id: mean.m,v 1.5 2009/02/08 18:35:21 sangwine Exp $

