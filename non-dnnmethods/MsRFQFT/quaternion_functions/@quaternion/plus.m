function q = plus(l, r)
% +   Plus.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2008, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% Three cases have to be handled:
%
% l is a quaternion, r is not,
% r is a quaternion, l is not,
% l and r are both quaternions.

% An additional complication is that the parameters may be empty (numeric,
% one only) or empty (quaternion, one or both). Matlab adds empty + scalar
% to give empty: [] + 2 gives [], not 2, but raises an error (Matrix
% dimensions must agree) on attempting to add empty to an array. This
% behaviour is copied here, whether the empty parameter is a quaternion or
% a numeric empty. The result is always a quaternion empty.

ql = isa(l, 'quaternion');
qr = isa(r, 'quaternion');

% Find the sizes and classes of the arguments or their components if they
% are quaternions. We avoid calling the quaternion size function because of
% the overhead. Instead we can directly compute the size using the x
% component.

if ql, ls = size(l.x); lc = class(l.x); ...
else   ls = size(l);   lc = class(l);      end
if qr, rs = size(r.x); rc = class(r.x); ...
else   rs = size(r);   rc = class(r);      end

pl = prod(ls); % The product of the elements of size(l).
pr = prod(rs); % The product of the elements of size(r).

sl = pl == 1; % Equivalent to isscalar(l), since ls must be [1,1]. Note 1.
sr = pr == 1; % Equivalent to isscalar(r).

el = pl == 0; % Equivalent to isempty(l), since ls must be [0,0], without
er = pr == 0; % calling an isempty function (quaternion or Matlab). Note 2.

% Now check for the case where one or other argument is empty and the other
% is scalar. In this case we choose to return an empty quaternion, similar
% to the behaviour of Matlab's plus function.

if (el && sr) || (sl && er)
    q = quaternion();
    return
end

% Having now eliminated the cases where one parameter is empty and the
% other is scalar, the parameters must now either match in size, or one
% must be scalar. It could be that they are both empty arrays of size
% [0, n] or [n, 0] etc, in which case we must return an empty array of the
% same size to match Matlab's behaviour.

eq = all(ls == rs); % True if l and r have the same size.

if ~(eq || sl || sr)
    error('Matrix dimensions must agree.');
end

if ~strcmp(lc, rc)
    error(['Class of left and right parameters does not match: ', ...
           lc, ' ', rc])
end

if el && er
   % We must return an empty (quaternion) matrix.
   if ql && qr
       q = l; % r would do just as well, since both are quaternions.
   elseif ql
       q = l; % This must be l because r isn't a quaternion.
   else
       q = r; % This must be r because l isn't a quaternion.
   end
   return
end

if ql && qr

    % The scalar part could be empty, and we have to handle this specially,
    % because [] + x is [] and not x, for some reason, in Matlab, as noted
    % above, whereas clearly if we add a pure quaternion to a full
    % quaternion, we want the result to have the scalar part of the full
    % quaternion and not be empty. Therefore we take care not to add the
    % empty scalar part, but treat it implicitly as zero, if one operand is
    % pure and the other is full.

    q = l;
    
    if isempty(q.w)
        q.w = r.w; % Copy the scalar part from r, may be empty.
    elseif ~isempty(r.w);
        q.w = q.w + r.w; % Add the scalar part of r, iff not empty.
    end

    q.x = q.x + r.x;
    q.y = q.y + r.y;
    q.z = q.z + r.z;

elseif isa(r, 'numeric')
    
    % The left parameter must be a quaternion, otherwise this function
    % would not have been called.

    q = l;
    if ~isempty(q.w)   
        q.w = q.w + r;
    else
        if eq
            q.w = r;
        else
            q.w = repmat(r, ls);
        end
    end

elseif isa(l, 'numeric')
    
    % The right parameter must be a quaternion, otherwise this function
    % would not have been called.
    
    q = r;
    if ~isempty(q.w)
        q.w = q.w + l;
    else
        if eq
            q.w = l;
        else
            q.w = repmat(l, rs);
        end
    end

else
  error('Unhandled parameter types in function +/plus')
end

% Note 1. Actually ls could be [1,1,1] or [1,1,1,...], but these cases are
% treated as scalar by Matlab (try it with rand(1,1,1) to see).

% Note 2. Actually ls could be [0,1,2,....], which is also empty. However,
% Matlab will add two arrays of this type of the same size to yield another
% empty array of the same size.

% $Id: plus.m,v 1.13 2010/03/22 19:28:26 sangwine Exp $
