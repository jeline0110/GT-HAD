function B = orthonormal_basis(V, W)
% ORTHONORMAL_BASIS creates an orthonormal basis from a pure quaternion V,
% and an optional pure quaternion W, which need not be perpendicular to V,
% but must not be parallel.
%
% The result is represented as a 3 by 3 orthogonal matrix, which may be
% complex if V and/or W are complex pure quaternions.

% Copyright © 2005, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isscalar(V)
    error('V must not be a vector or matrix.');
end

if ~isa(V, 'quaternion') || ~ispure_internal(V)
    error('V must be a pure quaternion.')
end

if nargin == 1
    W = orthogonal(V);
else
    W = orthogonal(V, W);
end

X = unit(cross(V, W)); % V may not have unit modulus at this point. 

% Ensure that V is a unit pure quaternion. We do this here and not before
% computing W and X, since orthogonal does not depend on V having unit
% modulus, and if we do give it unit modulus, it may become nilpotent in
% the complex case. However, we cannot leave it with non-unit modulus,
% otherwise B below would not be orthogonal.

V = unit(V);

B = [V.x, V.y, V.z; W.x, W.y, W.z; X.x, X.y, X.z];

if any(any(isnan(B) | isinf(B)))
    % TODO Better if we could detect this earlier and issue an error saying
    % that nilpotents make it impossible to construct a basis from V.
   error('QTFM:numeric', 'The basis matrix contains NaNs and/or Infs.') 
end

% It is possible for the result to be numerically inaccurate, and therefore
% we check that B is sufficiently orthogonal before returning. Error is
% particularly likely in the complex case if V and/or W is not defined
% accurately. (See the discussion note in the function unit.m.)

if norm(B * B.' - eye(3)) > 1e-12
    warning('QTFM:inaccuracy', ...
            'The basis matrix is not accurately orthogonal.');
end

% TODO: Consider an alternative method based on rotation of the standard
% basis. Given V, compute a rotation that rotates qi to V. Apply the same
% rotation to qj and qk, and you have a new basis which is bound to be
% orthogonal. The role of W in this needs to be thought out. Do we really
% need it? The rotation from V to qi must be computed around the axis
% perpendicular to the plane containing V and qi. This is easily done by
% computing q = scalar_product(V, qi) + vector_product(V, qi), then p =
% sqrt(q). The rotation is then conj(p).*qi.p to get V, and similarly with
% qj and qk to get the other two axes. This method won't work with
% nilpotents, but maybe this is not a limitation, since the method
% currently used also suffers from this problem.

% $Id: orthonormal_basis.m,v 1.10 2010/03/12 13:24:32 sangwine Exp $
