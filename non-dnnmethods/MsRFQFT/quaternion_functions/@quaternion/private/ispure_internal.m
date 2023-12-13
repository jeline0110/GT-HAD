function x = ispure_internal(q)
% Determines whether q is pure. This function has to know the
% representation of a quaternion, whereas the function ispure
% does not.

% Copyright © 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

x = isempty(q.w); % This is a call on the standard Matlab function
                  % isempty() since q.w is not a quaternion.
                  
% Implementation note added 28 June 2007.
%
% isempty(quaternion()) returns 1, because it checks only the w component.
% It is possible that this behaviour should be changed (perhaps to report
% an error) but this would require further study. The comments in ispure.m
% make clear that ispure(q) returns 1 if q is empty, and this is consistent
% with the statement that ispure(q) returns 1 if q has no scalar part.

% $Id: ispure_internal.m,v 1.6 2009/02/08 17:04:59 sangwine Exp $
