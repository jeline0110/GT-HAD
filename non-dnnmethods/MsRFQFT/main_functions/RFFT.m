function result =  RFFT(data,sigma)

addpath('.\ocf\')
tic
%% Band selection
dim = 6;
bandset = bandselect(data,dim);
FE = data(:,:,bandset);
for i = 1: dim
    FE(:,:,i) = mat2gray(FE(:,:,i));
end

%% Fourier transform & Filtering
FFT = fft2(FE);
Am = abs(FFT);
FL = FFT./Am;

Am=log(1+fftshift(Am));
Ak = imfilter(Am, fspecial('gaussian',[], sigma));

Ak=exp(ifftshift(Ak))-1;
FL_filted=Ak.*FL;
RestructedMap = abs(ifft2(FL_filted));

PhaseMap = abs(ifft2(FL));
PhaseMap = mat2gray(imfilter(PhaseMap.^2, fspecial('gaussian', [], sigma)));

%% Reconstruction Fusion
Fm = mean(RestructedMap,3);
Pm = mean(PhaseMap,3);
result = Fm.*Pm;
result = mat2gray(result);

%time = toc




