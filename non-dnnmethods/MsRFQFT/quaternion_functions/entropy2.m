
function e=entropy2(p);

 [H,W]=size(p);
%  z=p;
%  z=p/sum(p(:));
%  Ker=fspecial('gaussian',128,ceil(W/4));
%  Ker=Ker/max(Ker(:));
%  z=z.*Ker;
%  z=sum(z(:));
%W=128;
p=double(p);
p=removeNoise(p);% in order to avoid the boundary effect when computing entropy. We don't use this line in our paper.
%sgm=W*.01;
%sgm=W*.02;
 sgm=W*.03;
% You can set sgm=(0.01~0.03*W);
% As described in the paper, you should use a larger sgm if minimum region
% you concern is large; while you should use a smaller sgm if the minimum
% region you expecte to detect is small. In our paper, we set sgm=0.01*W in
% the predictin human fixations, and set sgm=0.03*W in predicting saliency
% regions human pay attention to.
p = imfilter(p, fspecial('gaussian',round(4*sgm),sgm));%Here you can use 3*sgm or 4*sgm
p=mat2gray(p);
p=p*255;
p=uint8(p);
e=entropy(p);
