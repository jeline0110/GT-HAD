
function  [SMs]=MSQF(f,M,N)
% Axf = axis(f);
% image(Axf);
mu = unit(quaternion(1,1,1));
FL =  qfft2(f, mu, 'L') ./ sqrt(M*N);
A=abs(FL);

%%
% P = angle(FL);
% Ax =axis(FL);
% figure,imshow(mat2gray(log(1+fftshift(A))))
% figure,colormap(jet); mesh(log(1+fftshift(A)))
% xlim([0 M]),ylim([0 N]),zlim([-5 10]),set(gca,'fontsize',20)

% figure,imshow(mat2gray(log(1+fftshift(P))))
% figure,image(fftshift(Ax)), axis off
%%
FL=FL./A;
A=log(1+fftshift(A));

for k=1:10
    Ak = imfilter(A, fspecial('gaussian',M,2^(k-2)));
%     if k==7
%         figure,colormap(jet); mesh(Ak)
%         xlim([0 M]),ylim([0 N]),zlim([-5 10]),set(gca,'fontsize',20)
%     end
    Ak=exp(ifftshift(Ak))-1;
    FL_filted=Ak.*FL;
    FIL = iqfft2(FL_filted, mu, 'L') .* sqrt(M*N);
    FIL=abs(FIL);
    FIL = mat2gray(FIL);
    SMs(:,:,k)=FIL.^2;
end