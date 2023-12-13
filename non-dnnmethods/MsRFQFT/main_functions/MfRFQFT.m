function result =  MfRFQFT(data,sigma)

addpath('.\ocf\')
addpath('.\quaternion_functions\')
addpath('.\LPP\');
addpath('.\OTVCA_V3\');

[nx, ny, nd] = size(data);

tic
datavec = reshape(data,nx*ny,nd);
%% Multi-feature extraction
dim = 3;
% OCF
bandset = bandselect(data,dim);
FE_OCF = data(:,:,bandset);
% PCA
[code] = pca(datavec');
FE_pca = reshape(code(:,1:dim),nx,ny,dim);
% OTVCA
FE_OTVCA = OTVCA_V3(data,dim);
% LPP
[mappedX, ~] = lpp(datavec, dim,10,1);
FE_lpp =reshape(mappedX,nx,ny,dim);

for i = 1: dim
    FE_OCF(:,:,i) = mat2gray(FE_OCF(:,:,i));
    FE_pca(:,:,i) = mat2gray(FE_pca(:,:,i));
    FE_OTVCA(:,:,i) = mat2gray(FE_OTVCA(:,:,i));
    FE_lpp(:,:,i) = mat2gray(FE_lpp(:,:,i));
end

%% Quaternion Fourier transform & Filtering
q_matrix = quaternion(FE_OCF,FE_pca,FE_OTVCA, FE_lpp);
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
result = mat2gray(result);

%time = toc




