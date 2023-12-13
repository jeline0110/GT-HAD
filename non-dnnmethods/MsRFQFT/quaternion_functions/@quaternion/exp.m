function Y = exp(X)
% EXP  Exponential
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% Modified 21 September 2005 to handle undefined axis correctly.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

if ispure_internal(X)
    
    % X is a pure quaternion (array).  We are computing exp(X) for each
    % element of the array, and we need to factor X to give a unit pure
    % quaternion, a, and a real or complex 'angle' denoted below by theta.
    % In fact this angle is simply the absolute value of X. We then have
    % X = a .* theta, and we can calculate exp(X) from cos(theta) +
    % a .* sin(theta) (de Moivre's formula).
    %
    % If the modulus of any element of X is zero or very small, the axis
    % (a) will be undefined, and we have to take care to handle this
    % without returning NaNs, since although the axis is undefined, the
    % exponential is not.
    
    theta     = abs(X);           % Since theta could be complex, we need
    undefined = abs(theta) < eps; % abs() again here to compare with eps.
    
    cc = cos(theta); % If theta is complex, the cosine and sine will yield
    ss = sin(theta); % cosh and I.*sinh of theta.

    theta(undefined) = 1; % This prevents divide by zero in the element
                          % positions where the axis is undefined.

    a = X ./ theta; % This is equivalent to axis(X) but without warnings
                    % about undefined axis because values of theta less
                    % than eps have been replaced with 1. This means the
                    % values at positions corresponding to undefined axis
                    % are incorrect, since we have copied the (small) value
                    % from X unmodified. However, these values will be
                    % multiplied by a zero or very small (less than eps)
                    % sine value below, so are harmless.

    Y = cc + a .* ss;
else
    
    % X is a full quaternion (array).  We use a recursive call to compute
    % the exponential of the vector part, and the standard Matlab function
    % to compute the exponential of the scalar part (which may be complex).
    % The result is the elementwise product of the two.

    Y = exp(s(X)) .* exp(v(X));
end

% $Id: exp.m,v 1.6 2009/02/08 18:35:21 sangwine Exp $

