function Dict=ConstructionD_lilu(data,K,P)
[D,N]=size(data);
IDX=kmeans(data',K,'start','plus');
Dict=[];
for i=1:K
    pos=find(IDX==i);
    D_temp=data(:,pos);
    if(size(D_temp,2)<P)
        continue;
    end
    mu=mean(D_temp,2);
    COV_inv=pinv(cov(D_temp'));
    D_temp_C=D_temp-repmat(mu,[1,size(D_temp,2)]);
    Dis=zeros(1,size(D_temp,2));
    for j=1:size(D_temp,2)
        Dis(j)=D_temp_C(:,j)'*COV_inv*D_temp_C(:,j);
    end
    [~,Ind]=sort(Dis);
    Dict=[Dict,D_temp(:,Ind(1:P))];
end
