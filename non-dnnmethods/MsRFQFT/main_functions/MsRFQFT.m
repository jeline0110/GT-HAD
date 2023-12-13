function result =  MsRFQFT(data,sigma)

addpath('.\ocf\')
addpath('.\quaternion_functions\')

[nx, ny, ~] = size(data);

tic
%% Multi-scale extraction
dim = 3; 
step = round(((nx+ny)/2)/10);
bandset{1} = bandselect(data,dim);
MS{1} = data(:,:,bandset{1});
MS{1} = imresize(MS{1},[nx, ny]+2*step);
for i = 1:3
    tmp = imresize(data,[nx, ny]-i*step);
    bandset{i+1} = bandselect(tmp,dim);
    MS{i+1} = tmp(:,:,bandset{i+1});
    MS{i+1} = imresize(MS{i+1},[nx, ny]+2*step);
end

for i = 1: dim
    MS{1}(:,:,i) = mat2gray(MS{1}(:,:,i));
    MS{2}(:,:,i) = mat2gray(MS{2}(:,:,i));
    MS{3}(:,:,i) = mat2gray(MS{3}(:,:,i));
    MS{4}(:,:,i) = mat2gray(MS{4}(:,:,i));
end

%% Quaternion Fourier transform & Filtering
q_matrix = quaternion(7*MS{1},MS{2},MS{3}, MS{4});
mu = unit(quaternion(1,1,1));
QFFT =  qfft2(q_matrix, mu, 'L');

Am=abs(QFFT);
FL=QFFT./Am; 

Am=log(1+fftshift(Am));
Ak = imfilter(Am, fspecial('gaussian',[], sigma));

Ak=exp(ifftshift(Ak))-1;
FL_filted=Ak.*FL;
RestructedMap = abs(iqfft2(FL_filted, mu, 'L'));

PhaseMap = abs(iqfft2(FL, mu, 'L'));
PhaseMap = mat2gray(imfilter(PhaseMap.^2, fspecial('gaussian', [], sigma)));

%% Reconstruction Fusion
Fm = mean(RestructedMap,3);
Pm = mean(PhaseMap,3);
result = Fm.*Pm;
result = imresize(result,[nx, ny]);
result = mat2gray(result);
%time = toc



