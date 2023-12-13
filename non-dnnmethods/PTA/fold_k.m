function T=fold_k(A,k,sizeT)
Num_post=1;
Num_pre=1;
Num=1;
Dim=length(sizeT);
for i=1:Dim
    Num=Num*sizeT(i);
    if i<k
        Num_pre=Num_pre*sizeT(i);
    end
    if i>k
        Num_post=Num_post*sizeT(i);
    end
end
T=zeros(Num,1);
num_b=Num_pre*sizeT(k);
for i=1:Num_post
    temp=A((i-1)*Num_pre+1:i*Num_pre,:);
    T((i-1)*num_b+1:i*num_b,1)=temp(:);
end
T=reshape(T,sizeT);

end