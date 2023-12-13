function B=CenterBias(A,CenterBiasValue)

[p,q]=size(A);
Mask=fspecial('gaussian',128,CenterBiasValue);
Mask=imresize(Mask,[p,q]);
Mask=Mask/max(Mask(:));

B=A.*Mask;