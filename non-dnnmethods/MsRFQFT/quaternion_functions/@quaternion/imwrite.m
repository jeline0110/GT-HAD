function imwrite(Q, varargin)
% IMWRITE  Write quaternion image array to graphics file.
% (Quaternion overloading of standard Matlab function.)
%
% This function takes the same parameters as the Matlab function IMWRITE,
% but the first must be a quaternion array, with elements of type uint8,
% uint16, or double. The type of image written to the file depends on the
% quaternion data in the array A, as follows. If the quaternion data has
% components of type uint16, then 16-bit samples will be written to the
% file. In all other cases 8-bit data will be written to the file. uint8 or
% uint16 pixel values are not scaled - they are written directly to the
% image file. double or single pixel values are assumed to be in the range
% [0, 1] and are scaled by 255 before writing to the file.

% Copyright © 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

if ~isa(Q, 'quaternion') 
    error(['Quaternion function IMWRITE called with non-quaternion',...
           ' as first parameter.'])
end

for k = 1:length(varargin)
    if isa(varargin{k}, 'quaternion')
        error('Only the first parameter is permitted to be a quaternion.')
    end
end

if ~ispure_internal(Q)
    error('Quaternion image array must be pure.')
end

C = class(exe(Q));
if strcmp(C, 'uint8') || strcmp(C, 'uint16') || strcmp(C, 'double')
    imwrite(cat(3, exe(Q), wye(Q), zed(Q)), varargin{:});
else
    error(['Quaternion array has unknown component class: ', C,...
           ' - expects uint8/16 or double'])
end

% $Id: imwrite.m,v 1.1 2009/03/02 15:12:03 sangwine Exp $
