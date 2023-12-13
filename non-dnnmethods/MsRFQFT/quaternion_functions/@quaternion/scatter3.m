function scatter3(Q, varargin)
% SCATTER3  Display pure quaternion array as 3D scatter plot
% (Quaternion overloading of standard Matlab function.)
%
% Takes the same parameters as the Matlab function of the same name, except
% that the first three parameters (X, Y, Z) are replaced by a single
% quaternion parameter, which must be a pure quaternion vector. For more
% sophisticated requirements, use the Matlab function and pass the X, Y and
% Z components of the quaternion array as the first three parameters.

% Copyright © 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargoutchk(0, 0, nargout))

for k = 1:length(varargin)
    if isa(varargin{k}, 'quaternion')
        error('Only the first parameter is permitted to be a quaternion.')
    end
end

if ~ispure_internal(Q)
    error('Quaternion array must be pure.')
end

if ~isvector(Q)
    error('Quaternion array must be a vector.')
end

scatter3(exe(Q), wye(Q), zed(Q), varargin);

% $Id: scatter3.m,v 1.1 2009/07/22 14:27:36 sangwine Exp $
