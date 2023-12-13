function C = conv2(w, x, y, z)
% CONV2 Two dimensional convolution.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function operates in a similar way to the standard Matlab function
% apart from supporting both left and right coefficients. (Since quaternion
% multiplication is not commutative, the general case requires both left
% and right multiplication in the convolution product/summation.) The
% Matlab function allows the first two parameters to be vectors - this is
% not implemented as yet. Acceptable calling profiles are:
%
% C = conv2(A, B)      - A is convolved on the left of B, that is A * B
% C = conv2({L, R}, B) - The convolution is L * B * R. L and R must be of
%                        the same size.
%
% The left and right coefficients must be the same size, unless one is a
% scalar, in which case it is promoted to the same size as the other.
%
% An optional last parameter can specify 'shape' as for the standard Matlab
% function. This is currently not implemented.

error(nargchk(2, 4, nargin)), error(nargoutchk(0, 1, nargout))

if nargin == 4
    error('conv2 does not yet support 4 parameters.')
end

% If two or three parameters are supplied, the first two can be either two
% matrices, or a cell array and a matrix. The cell array must contain two
% matrices of the same size, unless at least one is scalar, but they need
% not be quaternion valued, since they could be real or complex.
        
if iscell(w)
    % The first parameter must be a cell array of two elements, and the
    % elements must have the same size.
    
    if length(w) ~= 2
        error('First parameter must have two elements, if a cell array.')
    end
    
    L = w{1}; R = w{2};
    
    SL = isscalar(L);
    SR = isscalar(R);

    if ~(all(size(L) == size(R)) || SL || SR)
        error(['The elements of the cell array must have the same', ...
               ' size or at least one must be scalar.'])
    end
    
    if ~(isa(x, 'quaternion') || isnumeric(x))
        error('Second parameter must be quaternion or numeric.');
        % It can only be numeric if a third (quaternion) parameter is
        % passed, otherwise this function will not be called. Still better
        % to check it though, just in case.
    end
    
    B = x;
else
    % The first parameter is not a cell array. There must be two parameters
    % at least because we did a nargchk, so we check the first two here.
    
    % The first parameter must be a matrix (or vector). It need not be
    % quaternion-valued, since it could be a left matrix of coefficients.
    % We treat the smaller matrix as a coefficient matrix, but put it on
    % the left or right according to its position in the parameter list.
    % A scalar 1 is supplied for the coefficient matrix on the opposite
    % side.
    
    if ~(isa(w, 'quaternion') || isnumeric(w))
        error('First parameter must be quaternion or numeric.');
    end
    if ~(isa(x, 'quaternion') || isnumeric(x))
        error('Second parameter must be quaternion or numeric.');
    end

    if prod(size(w)) < prod(size(x)) % Is this the best comparison that
                                     % could be done? What if one or more
                                     % is long and thin? Should we use
                                     % max(size(w)) instead? TODO
        L = w; SL = isscalar(L);
        B = x;
        R = 1; SR = true;
    else
        L = 1; SL = true;
        B = w;
        R = x; SR = isscalar(R);
    end
end

if nargin == 3
    
    % The third parameter must be a 'shape' parameter. Check for the
    % legal possibilities.
    
    if ~ischar(y)
        error('Third parameter must be a string.');
    end

    if ~strcmp(y, 'full') && ~strcmp(y, 'same') && ~strcmp(y, 'valid')
        error(['The third parameter must be one of', ...
            '''full'', ''same'', or ''valid'''])
    else
        shape = y;
    end
else
    shape = 'full'; % This wasn't specified, so use a default value.
end

% TODO Because the shape parameters are not implemented, we check for
% values other than the default case, which we do handle.

if ~strcmp(shape, 'full')
    error('The shape parameter is not implemented.')
end

% The variables L, B, R now contain the matrices to be convolved, in the
% order L * B * R. L or R may be scalar, in which case SL and/or SR
% respectively will be true.

m1 = max(size(L, 1), size(R, 1)); % This is safe, because we have checked
m2 = max(size(L, 2), size(R, 2)); % that L and R are the same size or are
                                  % scalar. There is therefore no danger of
                                  % taking the first dimension from one
                                  % array and the second dimension from the
                                  % other.
[n1, n2] = size(B);

C = zerosq(m1 + n1 - 1, m2 + n2 - 1); % Preconstruct the result array.

if m1 * m2 < n1 * n2
    
    % Usual case, in which L/R are smaller than B and therefore we multiply
    % extracted elements of L/R by the whole of B.
    
    if SL && SR
        C = L .* B .* R; % A trivial 'convolution'.
    elseif SL
        for a = 1:m1
            for b = 1:m2
                %
                % C(a:a + n1 - 1, b:b + n2 - 1) = ...
                % C(a:a + n1 - 1, b:b + n2 - 1) + L .* B .* R(a, b);
                %
                T = substruct('()', {a:a + n1 - 1, b:b + n2 - 1});
                C = subsasgn(C, T, subsref(C, T) ...
                     + L .* B .* subsref(R, substruct('()', {a, b})));
            end
        end
    elseif SR
        for a = 1:m1
            for b = 1:m2
                %
                % C(a:a + n1 - 1, b:b + n2 - 1) = ...
                % C(a:a + n1 - 1, b:b + n2 - 1) + L(a, b) .* B .* R;
                %
                T = substruct('()', {a:a + n1 - 1, b:b + n2 - 1});
                C = subsasgn(C, T, subsref(C, T) ...
                   + subsref(L, substruct('()', {a, b})) .* B .* R);
            end
        end

    else
        % The normal case, in which L and R are vectors.
        for a = 1:m1
            for b = 1:m2
                %
                % C(a:a + n1 - 1, b:b + n2 - 1) = ...
                % C(a:a + n1 - 1, b:b + n2 - 1) + L(a, b) .* B .* R(a, b);
                %
                T = substruct('()', {a:a + n1 - 1, b:b + n2 - 1});
                U = substruct('()', {a, b});
                C = subsasgn(C, T, subsref(C, T)...
                   + subsref(L, U) .* B .* subsref(R, U));
            end
        end
    end
else
    % Unusual case in which B is smaller than L/R, so we extract elements
    % of B and multiply into the whole of L/R rather than the other way
    % around. We implement this because of the gross inefficiency of the
    % code above for the case where B is small and L/R are very much
    % larger.
    
    for a = 1:n1
        for b = 1:n2
            %
            % C(a:a + m1 - 1, b:b + m2 - 1) = ...
            % C(a:a + m1 - 1, b:b + m2 - 1) + L .* B(a, b) .* R;
            %
            T = substruct('()', {a:a + m1 - 1, b:b + m2 - 1});
            C = subsasgn(C, T, subsref(C, T) ...
                + L .* subsref(B, substruct('()', {a, b})) .* R);
        end
    end
end

% Implementation note:
%
% The method used above is due to Todd Ell, originally devised for the case
% of convolving/evaluating a linear quaternion system over a quaternion
% array (in the function conv2 of the LQSTFM toolbox -
% http://lqstfm.sourceforge.net/). See a detailed explanation in the file
% conv.m which uses the same method for one dimension.

% $Id: conv2.m,v 1.12 2009/02/08 18:35:21 sangwine Exp $

