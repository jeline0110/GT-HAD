function [tp, fp] = roc(t, y)
%
% ROC - generate a receiver operating characteristic curve
%
%    [TP,FP] = ROC(T,Y) gives the true-positive rate (TP) and false positive
%    rate (FP), where Y is a column vector giving the score assigned to each
%    pattern and T indicates the true class (a value above zero represents
%    the positive class and anything else represents the negative class).  To
%    plot the ROC curve,
%
%       PLOT(FP,TP);
%       XLABEL('FALSE POSITIVE RATE');
%       YLABEL('TRUE POSITIVE RATE');
%       TITLE('RECEIVER OPERATING CHARACTERISTIC (ROC)');
%
%    See [1] for further information.
%
%    [1] Fawcett, T., "ROC graphs : Notes and practical
%        considerations for researchers", Technical report, HP
%        Laboratories, MS 1143, 1501 Page Mill Road, Palo Alto
%        CA 94304, USA, April 2004.
%
%    See also : ROCCH, AUROC
%
% File        : roc.m
%
% Date        : Friday 9th June 2005
%
% Author      : Dr Gavin C. Cawley
%
% Description : Generate an ROC curve for a two-class classifier.
%
% References  : [1] Fawcett, T., "ROC graphs : Notes and practical
%                   considerations for researchers", Technical report, HP
%                   Laboratories, MS 1143, 1501 Page Mill Road, Palo Alto
%                   CA 94304, USA, April 2004.
%
% History     : 10/11/2004 - v1.00
%               09/06/2005 - v1.10 - minor recoding
%               05/09/2008 - v2.00 - re-write using algorithm from [1]
%
% Copyright   : (c) G. C. Cawley, September 2008.
%
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%
ntp = size(y,1);

% sort by classeifier output
[y,idx] = sort(y, 'descend');
t       = t(idx) > 0;
% generate ROC
P     = sum(t);
N     = ntp - P;
fp    = zeros(ntp+2,1);
tp    = zeros(ntp+2,1);
FP    = 0;
TP    = 0;
n     = 1;
yprev = -realmax;
for i=1:ntp
   if y(i) ~= yprev
      tp(n) = TP/P;
      fp(n) = FP/N; 
      yprev = y(i);
      n     = n + 1;
   end
   if t(i) == 1
      TP = TP + 1;
   else
      FP = FP + 1;
   end
end
tp(n) = 1;
fp(n) = 1;
fp    = fp(1:n);
tp    = tp(1:n);
% bye bye...
end
