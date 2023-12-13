function C = adjoint(A, F)
% ADJOINT   Computes an adjoint matrix of the quaternion matrix A. The
% adjoint matrix has 'equivalent' properties to the original matrix (for
% example, singular values).
%
% adjoint(A) or
% adjoint(A, 'complex')    returns a complex adjoint matrix.
% adjoint(A, 'real')       returns a real    adjoint matrix.
% adjoint(A, 'quaternion') returns a quaternion adjoint (A must be a
%                          biquaternion matrix in this case).
%
% The definition of the adjoint matrix is not unique (several permutations
% of the layout are possible).

% Copyright © 2005, 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 1
    F = 'complex'; % Supply the default parameter value.
end

if ~strcmp(F, 'real') && ~strcmp(F, 'complex') && ~strcmp(F, 'quaternion')
    error(['Second parameter value must be ''real'', ''complex''',...
           ' or ''quaternion''.'])
end

if strcmp(F, 'complex')

    % Reference:
    %
    % F. Z. Zhang, Quaternions and Matrices of Quaternions,
    % Linear Algebra and its Applications, 251, January 1997, 21-57.
    % DOI:10.1016/0024-3795(95)00543-9.
    % (The definition of the complex adjoint matrix is on page 29.)

    % Zhang's paper does not consider the case where the elements of the
    % quaternion are complex, but the adjoint definition is valid in this
    % case, although the four components of the quaternion are mixed among
    % the elements of the adjoint: this means that they must be unmixed by
    % the corresponding function unadjoint (using sums and differences).

    % Extract the components of A. We use scalar() and not s() so that we
    % get zero if A is pure.
    
    W = scalar(A); X = exe(A); Y = wye(A); Z = zed(A);

    C = [[ W + 1i .* X, Y + 1i .* Z]; ...
         [-Y + 1i .* Z, W - 1i .* X]];

elseif strcmp(F, 'real')
    
    % The definition of the real adjoint matrix (although not called that)
    % is given in several sources, but only for a singleton quaternion. The
    % extension to a matrix of quaternions follows easily from Zhang's
    % example in the complex case, and is easily checked to be valid. Up to
    % release 1.6 of QTFM the result returned was different, as shown in
    % the commented code to the right below (the first row and column were
    % transposed to obtain the new layout).
    
    % References:
    %
    % B. P. Ickes, 'A New Method for Performing Digital Control System
    % Attitude Computations using Quaternions', AIAA Journal, 8(1), January
    % 1970, pp13-17, American Institute of Aeronautics and Astronautics.
    % (Eqn 8 defines the layout used below.)
    %
    % Ward, J. P., 'Quaternions and Cayley numbers', Kluwer, 1997. (p91)
    %
    % Todd A. Ell, On Systems of Linear Quaternion Functions, February 2007
    % arXiv:math/0702084v1, http://www.arxiv.org/abs/math/0702084. (Eqn 3.)

    % Extract the components of A. We use scalar() and not s() so that we
    % get zero if A is pure.
    
    W = scalar(A); X = exe(A); Y = wye(A); Z = zed(A);

    C = [[W, -X, -Y, -Z]; ... % C = [[ W,  X,  Y,  Z]; ...  Code up to and
         [X,  W, -Z,  Y]; ... %      [-X,  W, -Z,  Y]; ...  including QTFM
         [Y,  Z,  W, -X]; ... %      [-Y,  Z,  W, -X]; ...  version 1.6.
         [Z, -Y,  X,  W]];    %      [-Z, -Y,  X,  W]];
     
else % F must be 'quaternion' since we checked it above.
    
    if isreal(A)
        error(['The ''quaternion'' adjoint is defined for', ...
               ' biquaternion matrices only'])
    end

    % Reference:
    %
    % Nicolas Le Bihan, Sebastian Miron and Jerome Mars,
    % 'MUSIC Algorithm for Vector-Sensors Array using Biquaternions',
    % IEEE Transactions on Signal Processing, 55(9), September 2007,
    % 4523-4533. DOI:10.1109/TSP.2007.896067.
    %
    % The quaternion adjoint is defined in equation 17 of the above paper,
    % on page 4525.
    
    RA = real(A); IA = imag(A);
    C = [RA, IA; -IA, RA];
   %C = [RA, IA; -IA, RA]; % Alternative code, note 2.
end

% Note: in the case of a biquaternion matrix, the three cases above do not
% cover all the possibilities, and the option strings are not accurately
% descriptive (since the 'real' adjoint actually has complex elements). Two
% more possibilities can be obtained by a double call on this function, and
% to make clear what the possibilities are, they are listed below:
%
% adjoint(b, 'complex') gives a 2x2 complex block per element of b. This is
% not simple as the elements of the adjoint are a mix of the elements of b,
% but it works.
%
% adjoint(b, 'real') gives a 4x4 complex block per element of b. In this
% case the values in the 4x4 block are copies of the four complex elements
% of b.
%
% adjoint(b, 'quaternion') gives a 2x2 quaternion block per element of b.
% This adjoint can be further processed (since the result is a quaternion
% matrix) to give two more adjoints of b, as listed below.
%
% adjoint(adjoint(b, 'quaternion'), 'complex') gives a 4x4 complex block
% per element of b, but not the same complex values as above - the real and
% imaginary parts are swapped around.
%
% adjoint(adjoint(b, 'quaternion'), 'real') gives an 8x8 real block per
% element of b, where the elements of the block are elements of b, but with
% changes of sign.
%
% The last two cases might merit further study.

% Note 2. The 'quaternion' adjoint requires further study. The alternative
% line of code at 103 above is under study - there are possible
% compatibility issues between the various different adjoints, particularly
% for biquaternions. A matching change is needed in unadjoint.m of course.

% Implementation note: in the case of a quaternion matrix (i.e. not a
% singleton quaternion), it is possible to construct the adjoint in two
% ways, quite apart from the permutations of layout mentioned above. These
% two ways are as above, with the quaternion matrix inserted in blocks into
% the adjoint (with negation and so on applied to the blocks); or in an
% interleaved form in which the blocks are distributed so that each
% quaternion is inserted into the adjoint as a block matrix following the
% pattern given above. For example, a quaternion matrix [q1, q2; q3, q4]
% could be assembled into an adjoint like this:
%
% [adjoint(q1), adjoint(q2); adjoint(q3), adjoint(q4)]
%
% An experiment will quickly show that the resulting adjoint has the same
% singular values as the adjoint constructed by this function, and that
% these are the same singular values as those of the original quaternion
% matrix.

% $Id: adjoint.m,v 1.11 2010/05/11 14:43:28 sangwine Exp $
