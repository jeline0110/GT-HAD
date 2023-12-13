function p = callHFTParams()
% Setting such a '.m' file is inspired by Harel's work.
p = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PostProcessin %%%%%%%%%%%%%%%%%%%%%%%%%%%

p.openBorderCut = 1;              % Turn border cut on (1) / off (0)
                                  % In our paper, we turn it off.
                                  % A border cut could be employ to alleviate the problem caused by the border effect. 
                                  % However the unfairness will be introduced when make comparison.
                                  
p.BorderCutValue = 0.01;          % Percentage of lines cut at each border.  BorderCutValue should be in (0,1].
                                  % This is valid when set p.openBorderCut = 1

p.setCenterBias = 1;              % Turn center bias on (1) / off (0)
                                  % In our paper, we turn it off.
                                  
p.CenterBiasValue = 42;           % The scale parameter of the centered Gaussian mask
                                  % This is valid when set p.setCenterBias = 1       
                                  
p.SmoothingValue = 0.04;          % final blur to apply to the saliency map
                                  % (in standard deviations of gaussian kernel,
                                  %  expressed as fraction of image width)
                                  % We set p.blurfrac = 0.05 in our paper. 
                                  
% Note that these posprocessing will influence the performance estimation based on ROC or other measure method.
% So calibration/compensation should be taken when make comparison. See more details in our paper.
                                  
                                   
