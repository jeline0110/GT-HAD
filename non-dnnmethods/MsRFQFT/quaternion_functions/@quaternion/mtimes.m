function q = mtimes(l, r)
% *   Matrix multiply.
% (Quaternion overloading of standard Matlab function.)
 
% Copyright © 2005, 2009 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

ql = isa(l, 'quaternion');
qr = isa(r, 'quaternion');

if ql && qr
    
    % Both arguments are quaternions. There are four cases to handle,
    % dependent on whether the arguments are pure or full.
    
    pl = ispure_internal(l);
    pr = ispure_internal(r);
    
    if pl && pr
        q = -mdot(l, r) + mcross(l, r);
    elseif pl
        q = l * r.w + l * vee(r);
    elseif pr
        q = l.w * r + vee(l) * r;
    else
        vr = vee(r);
        vl = vee(l);
        q = l.w * r.w + vl * r.w + ...
            l.w * vr  + vl * vr;
    end
   
else
    % One of the arguments is not a quaternion. If it is numeric, we can
    % handle it, and we must because the code above requires us to multiply
    % scalar parts by vector parts using a recursive call to this function.
    
    if ql && isa(r, 'numeric')
        if ispure_internal(l)
            q = class(compose(l.x * r, ...
                              l.y * r, ...
                              l.z * r), 'quaternion');
        else
            q = class(compose(l.w * r, l.x * r, ...
                              l.y * r, l.z * r), 'quaternion');
        end
    elseif isa(l, 'numeric') && qr
        if ispure_internal(r)
            q = class(compose(l * r.x, ...
                              l * r.y, ...
                              l * r.z), 'quaternion');
        else
            q = class(compose(l * r.w, l * r.x, ...
                              l * r.y, l * r.z), 'quaternion');
        end
    else
        error('Matrix multiplication of a quaternion by a non-numeric is not implemented.')
    end
end

% TODO Eliminate the recursive calls in this function. Profiling shows this
% function to be heavily called (as would be expected). The recursive calls
% mean that time is wasted on the isa checks at lines 8 and 9. About 10% of
% the time running the test suite is spent as self-time in this function.
% Some speedup has been achieved by using the dotted notation instead of
% private function calls.

% $Id: mtimes.m,v 1.6 2009/12/23 18:17:57 sangwine Exp $

