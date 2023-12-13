function A = dft_axis(RC)
% Private function to supply a default quaternion Fourier transform axis,
% real or complex depending on the (logical) parameter value. 1 indicates
% real, 0 complex.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(1, 1, nargout)) 

if ~islogical(RC)
    error('Argument must be a logical value');
end

% The value returned must be a root of -1. Real quaternion roots of -1 were
% studied by Hamilton and any unit pure real quaternion is a root of -1.
% The value chosen here is unbiased in any particular direction, since it
% has equal x, y, z components. Complex quaternion roots of -1 are not so
% simple. See the following for the theory behind the choice made here:
%
% S. J. Sangwine, Biquaternion (Complexified Quaternion) Roots of -1,
% Advances in Applied Clifford Algebras, 16(1), February 2006, 63-68.
% DOI:10.1007/s00006-006-0005-8.
%
% S. J. Sangwine, Biquaternion (complexified quaternion) roots of -1,
% June 2005. arXiv:math.RA/0506190, available at: http://arxiv.org/.

if RC

    % Real axis. We use the unit function because otherwise the result
    % would not have unit modulus.

    A = unit(quaternion(1,1,1));

else

    % Complex axis. This value has unit modulus as it stands and the real
    % and imaginary parts are perpendicular, both requirements for a root
    % of -1. Also, the modulus of the real part squared less the modulus
    % of the imaginary part squared, equals 1.

    A = complex(quaternion(1,1,1), quaternion(0,1,-1));

end

% $Id: dft_axis.m,v 1.6 2009/02/08 17:04:59 sangwine Exp $
