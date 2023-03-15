function [ s ] = tnorm( X, flag )
s=0;
if flag==1%% ell_1 norm
    s=sum(abs(X(:)));
else if strcmp(flag,'fro') %% F norm
        s=sum(sum(sum(X.*X)));
        s=s^0.5;
    end
end
        

end

