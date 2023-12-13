function r = cat(dim, varargin)
% CAT Concatenate arrays.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(3, inf, nargin)), error(nargoutchk(0, 1, nargout))

a = quaternion(varargin{1}); % The call to the class constructor ensures
b = quaternion(varargin{2}); % that a and b are both quaternions, even if
                             % the first or second varargin parameter was
                             % something else. Calling the constructor
                             % exploits the error checks there, that would
                             % be complex to include here for handling rare
                             % problems like catenating strings with
                             % quaternions.
if ispure_internal(a)
    if ispure_internal(b)
        % a and b are both pure, so the result will be too.
        r = class(compose(cat(dim, exe(a), exe(b)), ...
                          cat(dim, wye(a), wye(b)), ...
                          cat(dim, zed(a), zed(b))), 'quaternion');
    else
        % a is pure, but b isn't, so the result will need to be full and we
        % have to supply zeros for the scalar part of a with the same type
        % as the elements of a.
        t = exe(a);
        r = class(compose(cat(dim, zeros(size(a), class(t)), ess(b)),...
                          cat(dim, t,      exe(b)), ...
                          cat(dim, wye(a), wye(b)), ...
                          cat(dim, zed(a), zed(b))), 'quaternion');
    end
else
    if ispure_internal(b)
        % b is pure, but a isn't, so the result will need to be full and we
        % have to supply zeros for the scalar part of a with the same type
        % as the elements of a.
        t = exe(b);
        r = class(compose(cat(dim, ess(a), zeros(size(b), class(t))),...
                          cat(dim, exe(a), t     ), ...
                          cat(dim, wye(a), wye(b)), ...
                          cat(dim, zed(a), zed(b))), 'quaternion');
    else
        % a and b are both full quaternion arrays.
        r = class(compose(cat(dim, ess(a), ess(b)), ...
                          cat(dim, exe(a), exe(b)), ...
                          cat(dim, wye(a), wye(b)), ...
                          cat(dim, zed(a), zed(b))), 'quaternion');
    end
end

if nargin > 3
    r = cat(dim, r, varargin{3:end});    
end

% $Id: cat.m,v 1.6 2009/02/09 20:58:20 sangwine Exp $

