function [auc_pdpf,auc_pdtau,auc_pftau,PD,PF] = AUCall(result, map)

M = size(result,1)*size(result,2);
r0 = reshape(result,1,M);
mask = reshape(map, 1, M);
anomaly_map = logical(double(mask)>=1);
normal_map = logical(double(mask)==0);
r_max = max(r0(:));
taus = linspace(0, r_max, 10000);
for index2 = 1:length(taus)
    tau = taus(index2);
    anomaly_map_rx = (r0 > tau);
    PF(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
    PD(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end

auc_pdpf =  sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
auc_pdtau =  -(sum((taus(1:end-1)-taus(2:end)).*(PD(2:end)+PD(1:end-1))/2)); %sum((PD(1:end-1)-PD(2:end)).*(taus(2:end)+taus(1:end-1))/2);
auc_pftau =  -(sum((taus(1:end-1)-taus(2:end)).*(PF(2:end)+PF(1:end-1))/2));