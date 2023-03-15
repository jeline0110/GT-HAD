function [ data ] = NormalizeData( data )
%NORMALIZEDATA Summary of this function goes here
%   Detailed explanation goes here
[M N D]=size(data);
data=reshape(data,[M*N D]);
data=scale_new(data);
data=reshape(data,[M N D]);
end

