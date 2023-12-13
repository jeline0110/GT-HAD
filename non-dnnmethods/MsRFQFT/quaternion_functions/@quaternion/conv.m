function C = conv(A, B)
% CONV Convolution and polynomial multiplication.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function operates in the same way as the standard Matlab function
% apart from supporting both left and right coefficients. (Since quaternion
% multiplication is not commutative, the general case requires both left
% and right multiplication in the convolution product/summation.) To supply
% left and right coefficients, use the calling profile conv({L,R},v) where
% L and R are vectors of the same length and orientation, unless scalar. If
% the first parameter is not a cell array, it is taken to be a left
% coefficient and the right coefficient array is implicitly ones. The left
% and right coefficients must be the same size, unless one is a scalar, in
% which case it is implicitly promoted to the same size as the other.

error(nargchk(2, 2, nargin)), error(nargoutchk(0, 1, nargout))

if iscell(A)
    
    % The first parameter must be a cell array of two elements, and the
    % elements must have the same size (therefore orientation), unless one
    % is a scalar.
    
    if length(A) ~= 2
        error('If a cell array, the first parameter must have 2 elements.')
    end
    
    L = A{1}; R = A{2}; % Extract two vectors from the cell array.
    
    SL = isscalar(L);
    SR = isscalar(R);

    if ~(all(size(L) == size(R)) || SL || SR)
        error(['The elements of the cell array must have the same', ...
               ' size or at least one must be scalar.'])
    end
else
    % TODO: If only two parameters are given it would be better to assign
    % the longer of the two to B, and the shorter to L or R according as to
    % whether it should be multiplied on the left or right. But note that
    % we would have to keep a note of B's orientation... An alternative way
    % to achieve the same efficiency gain, would be to use a different
    % strategy in the loop, and extract singleton elements of B instead of
    % L and R.

    L = A; SL = isscalar(L);
    R = 1; SR = true; % Supply a default scalar value of 1 for R.
end

if ~isvector(L) || ~isvector(R) || ~isvector(B)
    % NB: The isvector function returns true for a scalar, so this error is
    % not triggered by one or more scalar parameters.
    error('Parameters must be vectors.')
end

LS = size(L);
RS = size(R);

assert(all(LS == RS) || SL || SR, ...
    'Left and right quaternion coefficients have different size');

LRS = max(LS, RS); % This gives the size and hence orientation of the left
                   % and right coefficients. (Either they have the same
                   % size or one is a scalar.)
                   
clear LS RS % Ensure we do not use them again, because we must use LRS.

m = prod(LRS); % The length of L/R, whichever (or both) is not scalar.
n = length(B);

% Make a zero result vector C, with the same orientation as L and/or R.

if LRS(1) == 1
     C = zerosq(1, m + n - 1); % L or R or both is/are a row vector.
else
     C = zerosq(m + n - 1, 1); % L or R or both is/are a column vector.
end

% To ensure that we can multiply L .* B .* R we need them to have the same
% orientation (row, column). Because we need the original orientation of B
% later, we keep this in a Boolean flag, before we modify B itself.

BS = size(B);

B_was_a_row_vector = BS(1) == 1;

if ~same_orientation(LRS, BS), B = B.'; end

% We are now ready to compute the convolution itself.  The algorithm used
% here is due to Todd Ell (see implementation note below for full details).
% The subsref/substruct/subasgn calls are needed because this is a class
% method and normal index notation cannot be used.

if m < n
    
    % Usual case, in which L/R are shorter than B and therefore we multiply
    % singleton elements of L/R by the whole of B.
    
    if SL && SR
        C = L .* B .* R; % A trivial 'convolution'.
    elseif SL
        for a = 1:m
            % C(a:a + n - 1) = C(a:a + n - 1) + L .* B .* R(a);
            T = substruct('()', {a:a + n - 1});
            C = subsasgn(C, T, subsref(C, T) + ...
                 L .* B .* subsref(R, substruct('()', {a})));
        end
    elseif SR
        for a = 1:m
            % C(a:a + n - 1) = C(a:a + n - 1) + L(a) .* B .* R;
            T = substruct('()', {a:a + n - 1});
            C = subsasgn(C, T, subsref(C, T) + ...
                 subsref(L, substruct('()', {a})) .* B .* R);
        end
    else
        % The most usual case, in which L and R are vectors.
        for a = 1:m
            % C(a:a + n - 1) = C(a:a + n - 1) + L(a) .* B .* R(a);
            T = substruct('()', {a:a + n - 1});
            U = substruct('()', {a});
            C = subsasgn(C, T, subsref(C, T) + ...
                 subsref(L, U) .* B .* subsref(R, U));
        end
    end
else
    % Unusual case in which B is shorter than L/R, so we extract singleton
    % elements of B and multiply into the whole of L/R rather than the
    % other way around. We implement this because of the gross inefficiency
    % of the code above for the case where B is short and L/R are very much
    % longer.
    
    for a = 1:n
        % C(a:a + m - 1) = C(a:a + m - 1) + L .* B(a) .* R;
        T = substruct('()', {a:a + m - 1});
        C = subsasgn(C, T, subsref(C, T) + ...
            L .* subsref(B, substruct('()', {a})) .* R);
    end
end

% Orient the result vector to match the orientation of the longer of L/R or
% B in order to return a result compatible with the standard Matlab conv
% function.

if m > n % L/R is longer than B.
    if ~same_orientation(LRS, size(C)), C = C.'; end
end
if m < n % B is longer than L/R.
    if B_was_a_row_vector ~= (size(C, 1) == 1), C = C.'; end
end
if m == n
    % The two parameters are of equal length.  In this case Matlab's conv
    % function seems to return a column vector unless both parameters are
    % row vectors.
    
    if LRS(1) == 1 && B_was_a_row_vector
        % L/R is/are row vectors and B was a row vector, so the result must
        % be a row vector.
        if size(C, 1) ~= 1, C = C.'; end
    else
        % The result must be a column vector, because L/R is/are not a row
        % vector or B was not.
        if size(C, 2) ~= 1, C = C.'; end
    end
end

end

function tf = same_orientation(U, V)
% Compares two vectors for orientation, returns true if they are both row
% or both column vectors, false otherwise. Call with size(A), size(B) as
% parameters, that is the two arguments must be of the form [r,c].

tf = any(U == V); % Since U = [1, x] or [x, 1] and V = [1, y] or [y, 1] if
                  % the vectors represented have the same orientation, then
                  % there must be a 1 in a matching position. If x == y
                  % these will also match, but no matter, because the
                  % orientations must be the same if the ones match.
end

% Implementation note:
%
% The method used above is due to Todd Ell, originally devised for the case
% of convolving/evaluating a linear quaternion system over a quaternion
% array (in the function conv2 of the LQSTFM toolbox -
% http://lqstfm.sourceforge.net/). Here, the left and right coefficient
% vectors L and R are split into singleton elements, and each pair of
% singleton elements is multiplied pointwise with the whole operand vector
% B in one vectorised operation. The result is then added to a slice of the
% output vector, the indices of this slice being determined by the indices
% of the singleton element of the left coefficient vector. This can be made
% clearer by a simple example (with no right coefficients). Take the left
% vector to be [1 2 3] and the operand vector to be [4 5 6 7 8]
% (one-dimensional for simplicity with real integer values). Then the
% convolution is computed by the method below using three intermediate
% pointwise products made from elements of the left array with the whole of
% the operand array, B:
%
% Y1 = 1 .* [4 5 6 7 8] = [ 4  5  6  7  8]
% Y1 = 2 .* [4 5 6 7 8] = [ 8 10 12 14 16]
% Y3 = 3 .* [4 5 6 7 8] = [12 15 18 21 24]
%
% and the results on the right are added together like this:
%
% Y1 = [ 4  5  6  7  8]
% Y2 =    [ 8 10 12 14 16]
% Y3 =       [12 15 18 21 24]
% ---------------------------
% +  = [ 4 13 28 34 40 37 24]
%
% exactly as returned by conv([1 2 3], [4 5 6 7 8]). Incidentally, this can
% be described in terms of linear filtering by regarding the intermediate
% output values Y1, Y2, and Y3 as the scaled, delayed impulse responses of
% the filter with impulse response [4 5 6 7 8]. Each scaling is due to the
% amplitude of the input sample, and the delay is due to the delay of the
% input sample relative to the start of the left sequence.

% $Id: conv.m,v 1.10 2009/02/08 18:35:21 sangwine Exp $

