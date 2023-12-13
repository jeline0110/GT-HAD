function n = numel(q, varargin)
% NUMEL   Number of elements in an array or subscripted array expression.
% (Quaternion overloading of standard Matlab function.)
%
% NOTE THAT this function does not return the number of quaternions in the
% array q. This can be obtained easily by prod(size(q)). The reason for
% this is complex, but connected with the way Matlab implements subscripted
% indexing. See the notes in the Matlab documentation for the numel
% function. (This function was changed in May 2008. Prior to this date, it
% did return prod(size(q)).)

% Copyright © 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

if isempty(q)
    n = 0; return % Return zero for an empty quaternion array.
end

if nargin == 1
    n = builtin('numel', q);
else
    % We have more than one argument. This means varargin represents a
    % list of index values. See the Matlab help documentation on the
    % numel function for a detailed (if unclear!) explanation of what
    % numel has to do. It appears that this function should never be
    % called with this parameter profile (Matlab should call the built-in
    % numel function instead), so we trap any call that does occur.

    error('Quaternion numel function called with varargin, unexpected.');

    % If we do have to handle this case, here is how it could be done:
    % n = numel(exe(q), varargin);
end

% $Id: numel.m,v 1.7 2009/02/08 18:35:21 sangwine Exp $

