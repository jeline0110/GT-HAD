function Y = sign(X)
% SIGN   Signum function
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function is equivalent to unit(X), but we implement it by calling
% the Matlab function using an isomorphic complex number. In the case where
% X is a complex quaternion we have to resort to direct coding, and we call
% the quaternion unit function. We could choose to use unit(X) in both
% cases, in which case sign(X) would simply be an alias for unit(X).

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if isreal(X)
    
    % X is a real quaternion, and we compute the signum of an isomorphic
    % complex number using the standard Matlab sign function, then
    % construct a quaternion with the same axis as the original quaternion.
    
    Y = isoquaternion(sign(isocomplex(X)), X);
else
    
    % X is a complex quaternion, and therefore we cannot use the method
    % above for real quaternions, because it is not possible to construct
    % an isomorphic complex number. We use instead the quaternion unit
    % function.
    
    Y = unit(X);
end;

% Note that a fundamental design feature of the QTFM toolbox is that it
% should implement all the standard Matlab functions and operators that
% have meaning in both the complex and quaternion cases, so that complex
% code can be easily adapted to work with quaternion matrices. This is why
% we provide the sign function, even though it has a synonym in the unit
% function. (Why we provide the unit function is another issue.)

% $Id: sign.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

