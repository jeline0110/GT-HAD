function [data, map] = ReadData(data_name)
%READDATA Summary of this function goes here
%   Detailed explanation goes here

switch lower(data_name)
    case '1'
    load('./data/SanDiego');
    case '2'
    load('./data/HYDICE');
    case '3'
    load('./data/Segundo');
    case '4'
    load('./data/GrandIsle');
end
end

