function R = blkdiag(varargin)
% BLKDIAG Construct block diagonal matrix from input arguments.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargoutchk(0, 1, nargout))

% Extract the components of the vector part of each input argument. The
% results here are cell arrays of the same length as varargin.

X = cellfun(@exe, varargin, 'UniformOutput', false);
Y = cellfun(@wye, varargin, 'UniformOutput', false);
Z = cellfun(@zed, varargin, 'UniformOutput', false);

if all(cellfun(@ispure_internal, varargin));
    
    % None of the input arguments is a full quaternion, so we can return a
    % pure quaternion result and ignore the W or scalar component.
    
    R = class(compose(blkdiag(X{:}), ...
                      blkdiag(Y{:}), ...
                      blkdiag(Z{:})), 'quaternion');
else
    
    % At least one of the input arguments is a full quaternion, so we need
    % to return a full quaternion result. We use the scalar function to
    % extract the W or scalar component, because this supplies a zero
    % result for a pure argument and thus avoids us having to check the
    % arguments individually.
    
    W = cellfun(@scalar, varargin, 'UniformOutput', false);
    
    R = class(compose(blkdiag(W{:}), ...
                      blkdiag(X{:}), ...
                      blkdiag(Y{:}), ...
                      blkdiag(Z{:})), 'quaternion');
end

% $Id: blkdiag.m,v 1.2 2009/02/08 18:35:21 sangwine Exp $

