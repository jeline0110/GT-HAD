function C = bsxfun(fun, A, B)
% BSXFUN  Binary Singleton Expansion Function
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(3, 3, nargin)), error(nargoutchk(0, 1, nargout))

if ~isa(fun, 'function_handle')
    error('First parameter must be a function handle')
end

F = func2str(fun); % String giving the name of the function fun.

if strcmp(F, 'plus') || strcmp(F, 'minus')
    
    % In these cases, bsxfun can be simply implemented by applying the
    % Matlab function bsxfun to the components of the quaternion arrays and
    % constructing the result from the results on the components. But the
    % code handles only the simplest case, so we need to check for other
    % possibilities first.
    
    if ~isa(A, 'quaternion') || ~isa(B, 'quaternion')
        error('quaternion/bsxfun cannot handle non-quaternion arguments');
    end
    
    if ispure(A) ~= ispure(B)
        error('quaternion/bsxfun cannot handle mixed pure and non-pure');
    end
    
    C = class(compose(bsxfun(fun, A.w, B.w), ...
                      bsxfun(fun, A.x, B.x), ...
                      bsxfun(fun, A.y, B.y), ...
                      bsxfun(fun, A.z, B.z)), 'quaternion');
         
elseif strcmp(F, 'eq') || strcmp(F, 'ne')
    
    % TODO These cases can probably be handled, but the result will be four
    % arrays of logicals which we need to combine into one logical array.
    
    error(['quaternion/bsxfun called with handle to function ', ...
           F, ' - bsxfun not yet implemented for this function'])
else
    
    % No other cases are currently handled, so we just raise an appropriate
    % error message.
    
    error(['quaternion/bsxfun called with handle to function ', ...
           F, ' - bsxfun not implemented for this function'])
end

% Implementation note.
%
% The reason for this function is that the Matlab functions var, cov and
% std call bsxfun. The early releases of QTFM worked with these functions
% at the time, but the Matlab code has been changed since then, and without
% an implementation of bsxfun for quaternion parameters, the var, cov and
% std functions will not work. The implementation above seeks to cover only
% the special cases needed to support these three Matlab functions. More
% cases may be added as the need arises.

% $Id: bsxfun.m,v 1.1 2009/03/08 18:12:19 sangwine Exp $
