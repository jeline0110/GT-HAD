function str = char(q)
% CHAR Create character array (string).
% (Quaternion overloading of standard Matlab function.)

% Note: the Matlab char function converts arrays of numeric values into
% character strings. This is not what this function does, but the Matlab
% guidance on user-defined classes suggests writing a char function and
% a disp/display function. This advice has been followed.

% Copyright © 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout)) 

% There are three cases to be handled. The argument is one of: empty, a
% pure quaternion, a full quaternion.

if isempty(q)
    str = '[] quaternion'; % This case must be handled first because an
    return;                % empty quaternion is not scalar, and the
end                        % next check would fail.

if ~isscalar(q)
    error('char cannot handle a vector or a matrix.')
end

function S = plusminus(X)
% Extracts the sign of X and returns the character '-', '+'.

if sign(X) == -1, S = '-'; else S = '+'; end
end

f = '%.4g'; % Control over the format of each numeric value.

if ~ispure_internal(q)
   % There is a scalar part to output:
   t = ess(q);
   if isreal(t)
       % The scalar part is real. A minus sign will be provided by num2str,
       % if needed.
       
       str = [num2str(t, f) ' '];
   else
       % The scalar part is complex, so just output it in parentheses, with
       % no sign (the real and imaginary parts will have a sign supplied by
       % num2str).
       
       str = ['(' num2str(t, f) ') '];
   end
else
   str = '';
end

% Now output the X/Y/Z components, with I/J/K and suitable signs and
% spaces. If the quaternion is pure, the X component is not preceded by a
% scalar part, and we treat it differently to avoid a space between the
% sign and the numeric value.

t = exe(q);
if ispure_internal(q)
    % No scalar part exists, so the X component is the first piece to be
    % output.
    
    if isreal(t)
        str = [num2str(t, f) ' * I '];
    else
        str = ['(' num2str(t, f) ') * I '];
    end
else
    % There was a preceding scalar part, so we have to space accordingly.
    
    if isreal(t)
        str = [str plusminus(t) ' ' num2str(abs(t), f) ' * I '];
    else
        str = [str '+ (' num2str(t, f) ') * I '];
    end
end

t = wye(q);
if isreal(t)
    str = [str plusminus(t) ' ' num2str(abs(t), f) ' * J '];
else
    str = [str '+ (' num2str(t, f) ') * J '];
end

t = zed(q);
if isreal(t)
    str = [str plusminus(t) ' ' num2str(abs(t), f) ' * K'];
else
    str = [str '+ (' num2str(t, f) ') * K'];
end
end

% $Id: char.m,v 1.10 2009/02/08 18:35:21 sangwine Exp $

