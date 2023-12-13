function X = cumprod(A, dim)
% CUMPROD   Cumulative product.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function cannot be implemented using the Matlab cumprod function,
% which is a built-in function, because a quaternion product cannot be
% computed componentwise.

% In what follows, we need to use subscripted references, but these do not
% work inside a class method (which this is). See the file 'implementation
% notes.txt', item 8.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout)) 

if nargin == 1
    
    % There is no dim parameter, so if the input is a vector, we can just
    % multiply all the elements together, if it is a matrix or array, we
    % need to work on the columns,....
    
    % The result has the same size as the argument, so we start by making a
    % copy. However, since A could be a pure quaternion array, and the
    % result cannot be pure in general, we must force X to be a full
    % quaternion array. A simple way to do this is to add zeros (to the
    % scalar part). The class function here is used to ensure that the
    % zeros have the same type as the components of A, so that if, for
    % example, A is a quaternion with int8 components, the array of zeros
    % will be of type int8.
    
    X = zeros(size(A), class(exe(A))) + A;
               
    % We then have to work along the copy, at each index value multiplying
    % on the left by the product of all the previous elements, which we
    % keep in P, to simplify the subsref/subsasgn calls.
    
    if isvector(A)
        
        P = subsref(X, substruct('()', {1})); % P = X(1);
        
        for k = 2:length(A)
            S = substruct('()', {k});
            P = P .* subsref(X, S); % P = P .* X(k);
            X = subsasgn (X, S, P); % X(k) = P;
        end

    elseif ndims(A) == 2 % A is a matrix. The algorithm is as above for
                         % vectors, but instead of a single element P, we
                         % now have a vector, with one element per column
                         % of A.

        P = subsref(X, substruct('()', {1, ':'})); % P = X(1,:);
        
        for k = 2:size(A, 1)
            S = substruct('()', {k, ':'});
            P = P .* subsref(X, S); % P = P .* X(k, :);
            X = subsasgn (X, S, P); % X(k, :) = P;
        end
        
    else % A has more than 2 dimensions.
        
        % Find the first non-singleton dimension and make a recursive call
        % with an explicit dimension parameter.
        
        d = size(A);
        for dim = 1:length(d)
            if d(dim) ~= 1
                X = cumprod(A, dim);
                return;
            end
        end
        
        % This should not happen, since if none of the dimensions is > 1,
        % the array should not have ndims > 2. But just in case ......
        
        error('There are no non-singleton dimensions to the array!')
    end
else
    if ~isnumeric(dim)
        error('Dimension argument must be numeric');
    end
    
    if ~isscalar(dim) || ~ismember(dim, 1:ndims(A))
        error(['Dimension argument must be a positive'...
               ' integer scalar within indexing range.']);
    end
    
    if ndims(A) == 2
        
        % A is a matrix and it can be handled by the code above. There are
        % two cases since the dim parameter can be only 1 or 2.
        
        if dim == 1
            
            % We are required to compute along the columns, which is the
            % default, so a plain recursive call will do the trick.
            
            X = cumprod(A);
        else
        
            % We are required to compute the product along the rows. We can
            % handle this using transpose, a recursive call to the code
            % above, and transposition of the result.

            X = cumprod(A.').';
        end
        return
    end
    
    % TODO: The more general case of multiple dimensions is more difficult.
    % See notes in the file prod.m, where the exact same problem arises.
        
    error('cumprod cannot yet handle arrays of dimension greater than 2.')    
end

% $Id: cumprod.m,v 1.5 2009/02/09 18:22:12 sangwine Exp $

