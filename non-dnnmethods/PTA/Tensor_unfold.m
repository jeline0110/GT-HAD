function out=Tensor_unfold(T,k)
sizeT=size(T);
NDim=length(sizeT);
Num=1;
Num_pre=1;
Num_post=1;
for i=1:NDim
    Num=Num*sizeT(i);
    if i<k
        Num_pre=Num_pre*sizeT(i);
    else if i>k
            Num_post=Num_post*sizeT(i);
        end
    end
end
out=zeros(Num/sizeT(k),sizeT(k));

B=reshape(T,[Num_pre,sizeT(k),Num_post]);
for i=1:sizeT(k)
    temp=squeeze(B(:,i,:));
    out(:,i)=temp(:);
end



end