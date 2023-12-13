function U = orthogonal(V, W)
% ORTHOGONAL(V) constructs a unit pure quaternion perpendicular to V, and
% perpendicular to W, if given. V (and W if given) must be pure quaternion
% scalar(s). W need not be perpendicular to V, but it must not be parallel
% (that is the cross product of W and V must not be zero or close to it).

% Copyright © 2005, 2010 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

error(nargchk(1, 2, nargin)), error(nargoutchk(0, 1, nargout))

if ~isscalar(V)
    error('First argument must not be a vector or matrix.');
end

if ~isa(V, 'quaternion') || ~ispure_internal(V)
    error('First argument must be a pure quaternion.')
end

if nargin == 2
    if ~isscalar(W)
        error('Second argument must not be a vector or matrix.');
        % TODO Remove this restriction. There is no reason why this
        % function could not be vectorised. However, there is no immediate
        % need, and this is likely to be a low priority task.
    end
    
    if ~isa(W, 'quaternion') || ~ispure_internal(W)
        error('Second argument must be a pure quaternion.')
    end

    if parallel(V, W)
        error('Arguments must not be parallel to each other.');    
    end
end

% Now compute the required pure quaternion.  The method used was published
% in:
% 
% Ell, T. A. and Sangwine, S. J., 'Decomposition of 2D Hypercomplex Fourier
% Transforms into Pairs of Complex Fourier Transforms', in: Gabbouj, M. and
% Kuosmanen (eds), 'Signal Processing X, Theories and Applications',
% Proceedings of EUSIPCO 2000, Tenth European Signal Processing Conference,
% Tampere, Finland, 5-8 September 2000, Vol. II, 1061-1064, section 6.
% 
% In the paper, a specific case was shown. Here we have a general method.
%
% Subsequent to the coding of this function, and the publication of the
% paper above by Ell and Sangwine, it was discovered that the algorithm had
% been published in 1957 in §5 of:
%
% J. K. Mackenzie and M. J. Thomson,  'Some Statistics Associated with the
% Random Disorientation of Cubes', Biometrika, 44(1-2), June 1957, 205-210.

if nargin == 1
    
    % W was not given, so we must choose an arbitrary vector not parallel
    % to V. We deal specially with the cases where V is in the set
    % {±qi, ±qj, ±qk} in order to return a sensible choice. For example, if
    % V == qi, we choose W so that the return result U == qj.

    L = parallel(V, [qi, qj, qk]);
    if any(L)

        % V is parallel to one of the standard basis vectors. We choose one
        % of the other two, cyclically. The negative signs ensure that, for
        % example, orthogonal(qi) yields +qj, as you would expect.
 
        if L(1)
            W = -qk;
        elseif L(2)
            W = -qi;
        else
            W = -qj;
        end
    else

        % V is not parallel to one of the standard basis vectors.
        
        W = quaternion(1, 1, 1); % The default, unless W == ±V.
        if parallel(V, W)
            % We have to make another choice in this case, somewhat
            % arbitrarily.
            W = quaternion(-1, -1, 0);
        end
    end
end

% Now compute the result.  We have to normalise this, because W may not be
% perpendicular to V, and therefore the result may not have unit modulus,
% even if W did. (This is why we do not normalise the values of W above:
% the normalization is done here to allow for the lack of perpendicularity,
% and it can therefore also adjust for the lack of unit modulus above.) It
% is possible that the call to unit( ) will fail if the cross product is a
% nilpotent biquaternion (whether this can happen is not known).

C = cross(V, W);
U = unit(C);

end

function tf = parallel(x, y)
% Returns true if x and y are parallel or 'close' to it. x and/or y may be
% complex pure quaternions, which is why there are two calls on abs -- to
% ensure that the result is real before comparison with the threshold. Note
% that this test may give a false false result if x and/or y is a nilpotent
% biquaternion (that is has a vanishing modulus) - meaning that the return
% result of the function is false, even though x is not in fact parallel to
% y. This is because of the vanishing modulus of the nilpotent.
tf = abs(abs(cross(x, y))) < eps;
end

% Implementation notes: this function must be deterministic, so that it
% returns the same result for a given V every time it is called, otherwise
% the various fft functions that depend on this function would need to be
% changed to pass the orthornormal basis as a parameter. This means that we
% cannot choose W at random if only V is given.
%
% The case where V is a complex quaternion is treated the same way, but
% there can be problems with nilpotents. 

% $Id: orthogonal.m,v 1.13 2010/03/12 13:23:40 sangwine Exp $
