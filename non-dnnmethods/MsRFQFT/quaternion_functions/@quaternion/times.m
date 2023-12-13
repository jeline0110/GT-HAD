function q = times(l, r)
% .*  Array multiply.
% (Quaternion overloading of standard Matlab function.)
 
% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

ql = isa(l, 'quaternion');
qr = isa(r, 'quaternion');

if ql && qr
    
    % Both arguments are quaternions. There are four cases to handle,
    % dependent on whether the arguments are pure or full, the first three
    % being subsets of the full quaternion product.

    if ispure_internal(l)
        if ispure_internal(r)
            % Both arguments are pure quaternions.
            q = class(compose(...
                           -(l.x .* r.x + l.y .* r.y + l.z .* r.z),...
                                          l.y .* r.z - l.z .* r.y, ...
                           - l.x .* r.z              + l.z .* r.x, ...
                             l.x .* r.y - l.y .* r.x             ),...
                'quaternion');
        else
            % The right argument is full, but the left is pure.
            q = class(compose(...
                           -(l.x .* r.x + l.y .* r.y + l.z .* r.z),...
                             l.x .* r.w + l.y .* r.z - l.z .* r.y, ...
                           - l.x .* r.z + l.y .* r.w + l.z .* r.x, ...
                             l.x .* r.y - l.y .* r.x + l.z .* r.w),...
                'quaternion');
        end
    else
        if ispure_internal(r)
            % The left argument is full, but the right is pure.
            q = class(compose(...
                           - l.x .* r.x - l.y .* r.y - l.z .* r.z, ...
                l.w .* r.x              + l.y .* r.z - l.z .* r.y, ...
                l.w .* r.y - l.x .* r.z              + l.z .* r.x, ...
                l.w .* r.z + l.x .* r.y - l.y .* r.x             ),...
                'quaternion');
        else
            % Both arguments are full quaternions. The full monty.
            q = class(compose(...
                l.w .* r.w -(l.x .* r.x + l.y .* r.y + l.z .* r.z), ...
                l.w .* r.x + l.x .* r.w + l.y .* r.z - l.z .* r.y, ...
                l.w .* r.y - l.x .* r.z + l.y .* r.w + l.z .* r.x, ...
                l.w .* r.z + l.x .* r.y - l.y .* r.x + l.z .* r.w),...
                'quaternion');
        end
    end
   
else

    % One of the arguments is not a quaternion. If it is numeric, we can
    % handle it, and we must because the code above requires us to multiply
    % scalar parts by vector parts using a recursive call to this function.
    
    if ql && isa(r, 'numeric')
        if ispure_internal(l)
            q = class(compose(l.x .* r, l.y .* r, l.z .* r), 'quaternion');
        else
            q = class(compose(l.w .* r, ...
                              l.x .* r, l.y .* r, l.z .* r), 'quaternion');
        end
    elseif isa(l, 'numeric') && qr
        if ispure_internal(r)
            q = class(compose(l .* r.x, l .* r.y, l .* r.z), 'quaternion');
        else
            q = class(compose(l .* r.w, ...
                              l .* r.x, l .* r.y, l .* r.z), 'quaternion');
        end
    else
        error('Multiplication of a quaternion by a non-numeric is not implemented.')
    end
end

% $Id: times.m,v 1.8 2009/02/08 18:35:21 sangwine Exp $

