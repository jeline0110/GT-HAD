function r = subsasgn(a, ss, b)
% SUBSASGN Subscripted assignment.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

switch ss.type
case '()'
    if length(ss) ~= 1
        error('Multiple levels of subscripting are not supported for quaternions.')
    end
    
    if ~isa(a, 'quaternion')
        error('Left side must be a quaternion in subscripted assignment if right side is a quaternion.')    
    end

    if ~isa(b, 'quaternion')

           % Argument a (the left-hand side of the assignment) is a quaternion,
           % but b (the right-hand side) is not (it could be double, for example,
           % zero). To handle this case, we need to convert b to a quaternion.

           if ispure_internal(a)
               error('Cannot convert right-hand argument to a pure quaternion.');
           else
               if isnumeric(b)
                   r = subsasgn(a, ss, quaternion(b)); % Convert b to a quaternion (with implicit
                   return                              % zero vector part), and call recursively.
               else
                   error('Cannot convert non-numeric right-hand argument to a quaternion.');
               end
           end
    end
    
    if ispure_internal(a) ~= ispure_internal(b)
        error('Left and right sides in subscripted assignment must both be pure, or both full quaternions.')
    end
    
    % To implement indexed assignment, we separate the quaternion into components,
    % perform the assignments on the components, and then construct the quaternion
    % result from the components.
    
    if ispure_internal(a)
        xa = exe(a); ya = wye(a); za = zed(a);
        xb = exe(b); yb = wye(b); zb = zed(b);
        
        xa(ss.subs{:}) = xb;
        ya(ss.subs{:}) = yb;
        za(ss.subs{:}) = zb;
        
        r = class(compose(xa, ya, za), 'quaternion');
    else
        sa = ess(a); xa = exe(a); ya = wye(a); za = zed(a);
        sb = ess(b); xb = exe(b); yb = wye(b); zb = zed(b);
        
        sa(ss.subs{:}) = sb;
        xa(ss.subs{:}) = xb;
        ya(ss.subs{:}) = yb;
        za(ss.subs{:}) = zb;
            
        r = class(compose(sa, xa, ya, za), 'quaternion');
    end
case '{}'
    error('Cell array indexing is not valid for quaternions.')
case '.'
    error('Structure indexing is not implemented for quaternions.')
    %
    % Possible use of structure indexing is to implement the following
    % sorts of assignment:
    %
    % q.x = blah
    %
    % However, there are some issues to be considered before implementing
    % such a scheme. Would it work, for example, if q doesn't exist? What
    % should happen if q is pure and q.s = blah occurs? Should q become a
    % full quaternion, or should an error be raised? What about mixed use
    % of array indexing and structure indexing, e.g. q.x(:,1)? Would it
    % work for q.x = q.x .* 2 where the structure indexing occurs on the
    % right hand side as well as on the left. Guess: q.x on the right would
    % be handled by subsref, not subsassgn.
    %
    % (Notes added after discussion with Sebastian Miron, 12 July 2005.)
    %
otherwise
    error('subsassign received an invalid subscripting type.')
end

% $Id: subsasgn.m,v 1.3 2009/02/08 18:35:21 sangwine Exp $

