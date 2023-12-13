function t = cumsum(a, dim)
% CUMSUM  Cumulative sum of elements.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout)) 

if nargin == 1
    t = overload(mfilename, a);
else
    if ~isnumeric(dim)
        error('Dimension argument must be numeric');
    end
    
    if ~isscalar(dim) || ~ismember(dim, 1:ndims(a))
        error(['Dimension argument must be a positive'...
               ' integer scalar within indexing range.']);
    end

    t = overload(mfilename, a, dim);
end

% $Id: cumsum.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

