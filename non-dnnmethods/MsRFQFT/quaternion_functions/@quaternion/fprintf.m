function count = fprintf(FID, format, A)
% FPRINTF Write formatted data to file.
% (Quaternion overloading of standard Matlab function.)
%
% Only one quaternion argument is permitted, unlike the standard
% Matlab function. The FID parameter may be omitted, in which case
% the output is sent to the standard output.
%
% The output is formatted with one quaternion per line of output.
% Spaces are automatically inserted between each of the components
% of each quaternion, and a \n is automatically inserted after each
% quaternion value output. The format string supplied should therefore
% not include spaces or \n.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(2, 3, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 2
    
    % Only two parameters have been supplied. The missing one must
    % be the FID, since the other two cannot be omitted. Therefore
    % shuffle the parameters to the right, and supply the default
    % value for the FID (1 = screen).
    
    if ~ischar(FID)
        error('If only two parameters are given the first must be a format string.');
    end
    
    A      = format;
    format = FID;
    FID    = 1;
end

N = prod(size(A)); % The number of quaternions to be output.

% Output is done from a real array R using the standard Matlab fprintf. To construct
% R we have to interleave the components of the quaternion array.

A = A.';

if ispure_internal(A)
    N = N * 3;
    R(1 : 3 : N - 2) = x(A);
    R(2 : 3 : N - 1) = y(A);
    R(3 : 3 : N    ) = z(A);
    count = fprintf(FID, [           format ' ' format ' ' format '\n'], R);
else
    N = N * 4;
    R(1 : 4 : N - 3) = s(A);
    R(2 : 4 : N - 2) = x(A);
    R(3 : 4 : N - 1) = y(A);
    R(4 : 4 : N    ) = z(A);
    count = fprintf(FID, [format ' ' format ' ' format ' ' format '\n'], R);
end

% $Id: fprintf.m,v 1.5 2009/02/08 18:35:21 sangwine Exp $

