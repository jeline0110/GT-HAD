function Y = tanh(X)
% TANH   Hyperbolic tangent.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

Y = sinh(X) ./ cosh(X);

% Note. This may not be the best way to implement tanh,
% but it has the merit of simplicity and will work for
% all cases for which sinh and cosh work.
% $Id: tanh.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

