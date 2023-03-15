function rgb = ImGray2Pseudocolor(gim, map, n)
% IMGRAY2PSEUDOCOLOR transform a gray image to pseudocolor image
%   GIM is the input gray image data
%   MAP is the colormap already defined in MATLAB, for example:
%      'Jet','HSV','Hot','Cool','Spring','Summer','Autumn','Winter','Gray',
%      'Bone','Copper','Pink','Lines'
%   N specifies the size of the colormap 
%   rgb is the output COLOR image data
%
% Main codes stolen from:
%       http://www.alecjacobson.com/weblog/?p=1655
%       %% rgb = ind2rgb(gray2ind(im,255),jet(255));                      %
%                                                                           


[nr,nc,nz] = size(gim);
rgb = zeros(nr,nc,3);

if ( ~IsValidColormap(map) )
    disp('Error in ImGray2Pseudocolor: unknown colormap!');
elseif (~(round(n) == n) || (n < 0))
    disp('Error in ImGray2Pseudocolor: non-integer or non-positive colormap size');
else
    fh = str2func(ExactMapName(map));
    
    rgb = ind2rgb(gray2ind(gim,n),fh(n));
    rgb = uint8(rgb*255);
end

if (nz == 3)
    rgb = gim;
    disp('Input image has 3 color channel, the original data returns');
end

function y = IsValidColormap(map)

y = strncmpi(map,'jet',length(map)) | strncmpi(map,'hsv',length(map)) |...
    strncmpi(map,'hot',length(map)) | strncmpi(map,'cool',length(map)) |...
    strncmpi(map,'spring',length(map)) | strncmpi(map,'summer',length(map)) |...
    strncmpi(map,'autumn',length(map)) | strncmpi(map,'winter',length(map)) |...
    strncmpi(map,'gray',length(map)) | strncmpi(map,'bone',length(map)) |...
    strncmpi(map,'copper',length(map)) | strncmpi(map,'pink',length(map)) |...
    strncmpi(map,'lines',length(map));

function emapname = ExactMapName(map)

if strncmpi(map,'jet',length(map))
    emapname = 'jet';
elseif strncmpi(map,'HSV',length(map))
    emapname = 'HSV';
elseif strncmpi(map,'hot',length(map))
    emapname = 'hot';
elseif strncmpi(map,'cool',length(map))
    emapname = 'cool';
elseif strncmpi(map,'spring',length(map))
    emapname = 'spring';
elseif strncmpi(map,'summer',length(map))
    emapname = 'summer';
elseif strncmpi(map,'autumn',length(map))
    emapname = 'autumn';
elseif strncmpi(map,'winter',length(map))
    emapname = 'winter';
elseif strncmpi(map,'gray',length(map))
    emapname = 'gray';
elseif strncmpi(map,'bone',length(map))
    emapname = 'bone';
elseif strncmpi(map,'copper',length(map))
    emapname = 'copper';
elseif strncmpi(map,'pink',length(map))
    emapname = 'pink';
elseif strncmpi(map,'lines',length(map))
    emapname = 'lines';
end
