function [A, B] = cdpolar(q, L)
% CDPOLAR: Polar form inspired by the Cayley-Dickson construction of a
% quaternion from two complex numbers. A and B are complex numbers
% equivalent to q, such that: q = A .* exp(B * j) in mathematical notation.
% In fact to get this to work in QTFM, we must convert the complex numbers
% into isomorphic quaternions like this, using the quaternion qi, rather
% than the complex i as the complex root of -1:
%
% q = (real(A) + imag(A) .* qi) .* exp((real(B) .* imag(B) .* qi) .* qj)
%
% or by using the dc function (inverse of the cd function):
%
% q = dc(A) .* exp(dc(B) .* qj)
%
% The second (optional) parameter, must be a logical array the same size as
% the quaternion parameter. True values in the logical array indicate which
% elements of A must have their sign inverted. (The sign of A has an
% ambiguity which cannot be resolved without outside information and this
% outside information can be supplied optionally in L. If omitted, elements
% of A take a default sign.)

% Copyright © 2008, 2009, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% Reference:
%
% Stephen J. Sangwine and Nicolas Le Bihan,
% 'Quaternion polar representation with a complex modulus and
% complex argument inspired by the Cayley-Dickson form',
% Advances in Applied Clifford Algebras, 20 (1), March 2010, 111--120,
% doi:10.1007/s00006-008-0128-1.
% Also available as preprint arXiv:0802.0852, 6 February 2008, available at
% http://arxiv.org/abs/arxiv:0802.0852.
%
% (The paper does not explicitly cover the special case dealt with at lines
% 66-74 below, although, since A can be chosen arbitrarily when the (w, x)
% parts of q are near zero, the result produced here in that case is
% implicitly described in the paper.)

error(nargchk(1, 2, nargin)), error(nargoutchk(2, 2, nargout))

if ~isreal(q)
    error('cdpolar cannot handle complexified quaternions.')
end

if nargin == 2
    if ~islogical(L)
        error('Second parameter must be a logical array.')
    end
    if any(size(q) ~= size(L))
        error('Second parameter must have the same size as q.')
    end
end

% The first step in computing A and B is to compute a, which is a unit
% complex number formed from the scalar and X parts of q scaled by the
% modulus of q (the result may be zero, or close to it, which is handled as
% a special case).

[a, b] = cd(q); % Split q into two complex numbers using the standard
                % Cayley-Dickson form. a will be the (w, x) components,
                % b will be the (y, z) components.

A = abs(q) .* sign(a); % Calculate A. This is the default case. Note 1.

az = abs(a) ./ abs(b) < 1e-12; % Find the elements of a with small modulus
                               % relative to the modulus of b (or q). The
                               % threshold value is somewhat arbitrary.
if nnz(az) ~= 0
    % Some elements of a were zero, or close to it relative to b, and
    % therefore so are the corresponding elements of A. Replace these
    % elements with the value in the (y, z) part. Note 2.
    A(az) = b(az);
end

% Apply the sign correction to A, at selected elements, iff the second
% parameter was supplied.

if nargin == 2
    A(L) = -A(L); % Note 7.
end

% Now, since q is the product of A with exp(Bj), we can calculate exp(Bj)
% by dividing q on the left by A. Since left division is not implemented in
% QTFM (note 3) we actually multiply on the left by the inverse of the
% quaternion representation of A to get T, an intermediate result of unit
% modulus. (T must have unit modulus, because A has the modulus of q.)

T = conj(dc(A)) .* q ./ (real(A).^2 + imag(A).^2); % Note 4.

% From Lemma 1 in the paper referenced above, the X component of T is
% theoretically zero. We set this exactly to zero to avoid any further
% propagation of small non-zero values which may cause needless errors in
% the calculation of angle(T). We use class to ensure that the zeros have
% the same data type as the components of T (and q). At the same time, we
% initialise the result B to the same type of zeros, as we need default
% values for any elements that are not assigned below.

T.x = zeros(size(T), class(T.x)); B = T.x;

% Values in T may now be NaNs, if A has any elements with zero modulus. It
% is also possible for some values in T to have small vector parts. The
% NaNs are handled below after B is computed, but the small vector parts
% are dealt with next to avoid problems with inaccurate axis values. NZV is
% a logical array indicating which elements of T have non-zero vector
% parts. Elements of B not selected by NZV are returned as zero.

NZV = y(T).^2 + z(T).^2 > 1e-8; % Equivalent to modsquared(v(T)) > 1e-8.

% Because T has unit modulus, computing its log is a special case. Note 5.

TNZV = subsref(T, substruct('()', {NZV})); % Note 6.

B(NZV) = sign(complex(y(TNZV), z(TNZV))) .* angle(TNZV); % Note 7.

% Some elements of B may now be NaNs, because the corresponding value in Bj
% was zero. We don't want to return these, so we change them to zeros,
% provided the corresponding values of A are also zero or very small. This
% means that cdpolar behaves similarly to the Matlab function angle, which
% returns an angle of zero for a complex number which is zero or very
% small.

B(isnan(B) & (abs(A) < eps)) = 0;

% Notes:
%
% 1. The Matlab sign function applied to a complex value yields z./abs(z).
%
% 2. There is no need to use sign or abs here, because abs(q) and abs(b)
%    are almost the same (since abs(a) is very much less than abs(b)).
%
% 3. Type help quaternion/ldivide or see ldivide.m to see why.
%
% 4. real(A).^2 + imag(A).^2 is equal to modsquared(dc(A)) but is computed
%    on the two non-zero components of dc(A) only.
%
% 5. log(q) = log(abs(q)) + axis(q) .* angle(q). Since abs(q) = 1, and
%    log(1) = 0, we need only compute the axis times the angle.
%
% 6. Because this is a class method, we cannot use subscripted indexing on
%    T. We have to use subsref. See the file 'implementation notes.txt',
%    item 8 for more.
%
% 7. The RHS of this statement is equivalent to axis(TNZV) .* angle(TNZV).
%    Since the X component of T is exactly zero, we need only compute using
%    the Y and Z components. Therefore, instead of using quaternions and
%    the axis function, we can calculate B directly in complex arithmetic
%    using the Matlab sign function in place of axis (it yields the same
%    result, complete with inaccuracies for small moduli, which we deal
%    with using NZV).

% $Id: cdpolar.m,v 1.12 2010/05/11 14:43:49 sangwine Exp $
