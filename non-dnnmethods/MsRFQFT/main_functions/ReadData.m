function [data map] = ReadData(data_name)
%READDATA Summary of this function goes here
%   Detailed explanation goes here

switch lower(data_name)
    case '11'
    load('./data/abu-airport-1');
    case '12'
    load('./data/abu-airport-2'); 
    case '13'
    load('./data/abu-airport-3');
    case '14'
    load('./data/abu-airport-4');
    case '21'
    load('./data/abu-beach-1');
    case '22'
    load('./data/abu-beach-2'); 
    case '23'
    load('./data/abu-beach-3');
    case '24'
    load('./data/abu-beach-4');
    case '31'
    load('./data/abu-urban-1');
    case '32'
    load('./data/abu-urban-2'); 
    case '33'
    load('./data/abu-urban-3');
    case '34'
    load('./data/abu-urban-4');
    case '35'
    load('./data/abu-urban-5');
    case '4'
    load('./data/HYDICE_urban');
    case '5'
    load('./data/San_Diego');
    case '6'
    load('./data/D1F12H1');
    case '7'
    load('./data/Cooke_280X300');
    case '8'
    load('./data/Cooke');
    case '110'
    load('./data/Salinas_syn_1X1');
    case '111'
    load('./data/Salinas_syn_3X3');
    case '112'
    load('./data/Salinas_syn_5X5');
    case '113'
    load('./data/Salinas_syn_7X7');
    case '200'
    load('./data/salinas_impl14');
    case '201'
    load('./data/O_salinas');
end
end

