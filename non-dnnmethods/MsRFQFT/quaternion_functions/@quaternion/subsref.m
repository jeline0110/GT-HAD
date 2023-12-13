function b = subsref(a, ss)
% SUBSREF Subscripted reference.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

if length(ss) ~= 1
    error('Only one level of subscripting is currently supported for quaternions.');
    % See the notes below under structure indexing.
end

switch ss.type
case '()'
    if length(ss) ~= 1
        error('Multiple levels of subscripting are not supported for quaternions.')
    end
    
    % To implement indexing, we separate the quaternion into components,
    % and then compose the quaternion from the indexed components.
    
    if ispure_internal(a)
        xa = exe(a); ya = wye(a); za = zed(a);
        b = class(compose(xa(ss.subs{:}), ...
                          ya(ss.subs{:}), ...
                          za(ss.subs{:})), 'quaternion');
    else
        sa = ess(a); xa = exe(a); ya = wye(a); za = zed(a);
        b = class(compose(sa(ss.subs{:}), ...
                          xa(ss.subs{:}), ...
                          ya(ss.subs{:}), ...
                          za(ss.subs{:})), 'quaternion');
    end
case '{}'
    error('Cell array indexing is not valid for quaternions.')
case '.'
    % Structure indexing. TODO.
    %
    % See some notes on this subject in the file subsasgn.m. Here, we would
    % have to support two levels of indexing, such as q.x(1,2) and q(1,2).x
    % which would give the same result. The Matlab help on subsref explains
    % how this would work. We could accept s, x, y, or z as field names and
    % return x(a) etc, but we would really need to pass a second level of
    % subscripting back to subsref recursively. The code below is an
    % interim step to support one level of structure indexing.
    %
    % 15-Sep-2005 Code contributed by T.A. Ell.
    %
    switch ss.subs
        case {'vector', 'v'}    
            b = vee(a);
        case {'scalar', 's', 'w'} % w added as synonym for s, 23 May 2008.
            b = scalar(a); % Calling scalar() rather than s() means an
                           % array of zeros is returned if a is pure.
        case {'x', 'I'}
            b = exe(a);
        case {'y', 'J'}
            b = wye(a);
        case {'z', 'K'}
            b = zed(a);
        case 'imag'
            b = imag(a);
        case {'real'}
            b = real(a);
        otherwise
            error( ['Structure ''.', ss.subs, ''' is not a valid index']);
    end
otherwise
    error('subsref received an invalid subscripting type.')
end

% $Id: subsref.m,v 1.7 2009/02/08 18:35:21 sangwine Exp $

