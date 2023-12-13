function display(q)
% DISPLAY Display array.
% (Quaternion overloading of standard Matlab function.)

% Copyright © 2005, 2008 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

if length(inputname(1)) == 0
    
    name = 'ans';
    
    % TODO It would be good if we could update ans to have the value of q,
    % but even the standard Matlab display function doesn't do this.
    % For example, try: display(2 * pi). The result is:
    %
    % ans =
    % 
    %     6.2832
    %
    % but the variable ans is not modified if it exists, and not created if
    % it doesn't exist. So although what happens here is not ideal, it does
    % at least match what Matlab does.
else
    name = inputname(1);
end

if isequal(get(0,'FormatSpacing'),'compact')
  disp([name ' =']);
  disp(q);
else
  disp(' ');
  disp([name ' =']);
  disp(' ');
  disp(q);
  disp(' ');
end

% $Id: display.m,v 1.4 2009/02/08 18:35:21 sangwine Exp $

