function r = unit(a)
% UNIT quaternion. Divides a quaternion by its own modulus.
% The result is a quaternion with unit modulus.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

m = abs(a);

% Since m could be complex, the warning check below has to
% use abs again to get a real result for comparison with eps.

if any(any(abs(m) < eps))
    warning('QTFM:information', ...
            ['At least one element has zero modulus, '...
             'and divide by zero will occur.']);
end

r = a ./ m;
 
% Dividing by a small modulus can result in numerical errors such that the
% result does not have unit modulus. This is especially likely with complex
% quaternions, so we perform a check here.

% The modulus (and norm) of each element of r should be either 1 or 1 + 0i. 
% In either case, subtracting ones should result in a real or complex
% number with very small real and imaginary parts, which we compare with
% epsilon using an arbitrary scale factor which we choose so that the test
% is not too sensitive.

d = normq(r) - 1;
n = nnz(abs(real(d)) > 1e3.*eps | abs(imag(d)) > 1e3.*eps);

if n > 0
    if n == 1
        warning('QTFM:information', ...
            ['One element of the result of the unit '...
            'function has a modulus which is not accurately 1.']);
    else
        warning('QTFM:information', ...
            [num2str(n), ' elements of the result of the unit '...
            'function have a modulus which is not accurately 1.']);
    end
end

% Discussion note. Unit complex pure quaternions must have real and
% imaginary parts with moduli b and d such that b^2 - d^2 = 1. See:
%
% Sangwine, S. J., 'Biquaternion (Complexified Quaternion) Roots of -1',
% Advances in Applied Clifford Algebras, 16, (1), February 2006, 63-68.
% doi: 10.1007/s00006-006-0005-8, also available as arXiv:math.RA/0506190,
% 10 June 2005, available at http://www.arxiv.org/.
%
% It is possible for b and d to be large, and then the difference between
% their squares can be lost below the least significant bits. Therefore,
% although it is possible in theory to construct a unit pure complex
% quaternion by applying the unit function to an arbitrary pure complex
% quaternion, it is much better to create the unit pure complex quaternion
% accurately in the first place by choosing correct values for b and d.
%
%                                                     SJS 28 September 2005

% $Id: unit.m,v 1.9 2009/11/12 19:41:32 sangwine Exp $

