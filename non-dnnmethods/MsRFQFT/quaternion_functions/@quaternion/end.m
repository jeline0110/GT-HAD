function e = end(a, k, n)
% End indexing for quaternion arrays.

% Copyright © 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% This function is implemented by calling the Matlab builtin function on
% the x component of a. This means the complexities of implementing this
% function can be avoided, which is helpful given the scant information
% provided in the Matlab documentation (see Matlab -> programming ->
% Classes and Objects -> Designing User Classes in Matlab -> Defining end
% Indexing for an Object). Prior to 31 March 2008, this code was
% implemented incorrectly as e = size(a, k) which fails when a is a row
% vector. Note that we do not check n, because it is passed to the builtin
% function, and will presumably raise an error there if there is a problem.

e = builtin('end', exe(a), k, n);

% $Id: end.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

