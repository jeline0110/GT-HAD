function show_internal(name, value)
% Displays the four components of a quaternion (array). This code is called
% from show.m and displayall.m, which must pass the name of the variable as
% a string. (It may be empty if the value is anonymous, as result for
% example, from the call 'show(qi)'.)

% Copyright © 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

if isempty(value)
    
    if ~isempty(name)
        disp(' ');
        disp([name, ' =']);
    end

    % There are no numeric values to be output, so we simply output a
    % description of the empty value, depending on its size, which may be
    % 0-by-0, or 0-by-N, and it may be a matrix (2-dimensional) or an array
    % (more than two-dimensional). We can output something informative in
    % all these cases and not just '[]'.
    
    S = blanks(5);
    d = size(value);
    if sum(d) == 0
        S = [S '[] quaternion'];
    else
        S = [S 'Empty quaternion'];
        l = length(d);
        if l == 2
            S = [S ' matrix: '];
        else
            S = [S ' array: '];
        end
        for k = 1:l
            S = [S, num2str(d(k))];
            if k == l
                break % If we have just added the last dimension, no need
                      % for another multiplication symbol.
            end
            S = [S, '-by-'];
        end
    end
    disp(S)
else
    % The quaternion is not empty, therefore we can output numeric data. We
    % do this by outputting the three or four components one by one. There
    % is a special case if the quaternion is scalar, since we can output
    % this on one line, using disp.
    
    if isscalar(value)
        disp(' ');
        disp([name, ' =']);
        disp(' ');
        disp(value);
        disp(' ');
    else
        disp(' ');
        if ~ispure_internal(value)
            if isempty(name)
                disp('S ='); disp(' ');
            else
                disp([name '.S =']); disp(' ');
            end
            disp(s(value));
        end
        if isempty(name)
            disp('X ='); disp(' ');
        else
            disp([name '.X =']); disp(' ');
        end
        disp(x(value));
        if isempty(name)
            disp('Y ='); disp(' ');
        else
            disp([name '.Y =']); disp(' ');
        end
        disp(y(value));
        if isempty(name)
            disp('Z ='); disp(' ');
        else
            disp([name '.Z =']); disp(' ');
        end
        disp(z(value));
    end
end

% $Id: show_internal.m,v 1.2 2009/02/08 17:04:59 sangwine Exp $
