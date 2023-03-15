function [ L,E,rank] = dictionary_learning_tlrr( XX, opts)
[L,E,rank,obj,err,iter] = trpca_tnn_w(XX,opts.lambda,opts);
end

