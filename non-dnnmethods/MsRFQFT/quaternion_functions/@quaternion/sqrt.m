function Y = sqrt(X)
% SQRT   Square root.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2006 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

if isreal(X)
    
    % X is a real quaternion, and we compute the square root of an
    % isomorphic complex number using the standard Matlab square root
    % function, then construct a quaternion with the same axis as the
    % original quaternion.
    
    Y = isoquaternion(sqrt(isocomplex(X)), X);
else
    
    % X is a complex quaternion, and therefore we cannot use the method
    % above for real quaternions, because it is not possible to construct
    % an isomorphic complex number. Therefore we use polar form and halve
    % the argument. Note that the modulus and argument here are complex,
    % so the square root of the modulus is complex.
    
    % Preconstruct a result array of zero quaternions, of the same size as
    % X and with components of the same class.
    
    Y = zerosq(size(X), class(x(X)));
    
    % If the vector part of any element of X is zero, computing its axis
    % will result in an undefined axis warning. We avoid this warning by
    % computing just the square root of the scalar part in these cases.
    % There may be no such cases, or all the elements in X may have
    % undefined axis.
    
    undefined = abs(normq(v(X))) < eps;
    defined   = ~undefined;
        
    % In order to perform the subscripted assignment using the logical 
    % array undefined we have to use subsasgn and substruct because normal
    % indexing does not work here in a class method.
    % See the file 'implementation notes.txt', item 8, for more details.

    if nnz(undefined) > 0
        % There are some cases of undefined axis.
        U = scalar(subsref(X, substruct('()',  {undefined})));
        Y = subsasgn(Y, substruct('()', {undefined}), quaternion(sqrt(U)));
    end
    
    if nnz(defined) > 0
        % There are some cases where the axis is defined.
        D = subsref(X, substruct('()', {defined}));    
        Y = subsasgn(Y, substruct('()', {defined}), ...
                    sqrt(abs(D)) .* exp(axis(D) .* angle(D) ./ 2));
    end
end;

% TODO Consider whether a better algorithm could be devised for the complex
% case based on the Cartesian form. A formula for the complex square root
% is given as eqn 12 on page 17 of:
%
% John H. Mathews,
% 'Basic Complex Variables for Mathematics and Engineering',
% Allyn and Bacon, Boston, 1982. ISBN 0-205-07170-8.

% $Id: sqrt.m,v 1.5 2009/11/12 19:43:25 sangwine Exp $

