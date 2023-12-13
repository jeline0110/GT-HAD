function R = overload(F, Q, varargin)
% Private function to implement overloading of Matlab functions. Called to
% apply the function F to the quaternion array Q by operating on components
% of Q with F. F must be a string, giving the name of the function F. The
% calling function can pass this string using mfilename, for simplicity of
% coding. varargin contains optional arguments that are not quaternions.

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

H = str2func(F); % A handle to the function designated by F.

R = Q;

if ~isempty(R.w) % Skip the scalar part if it is empty (pure quaternion).
    R.w = H(R.w, varargin{:});
end

R.x = H(R.x, varargin{:});
R.y = H(R.y, varargin{:});
R.z = H(R.z, varargin{:});

% $Id: overload.m,v 1.3 2009/02/08 17:04:59 sangwine Exp $
