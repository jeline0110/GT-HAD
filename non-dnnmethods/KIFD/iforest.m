function scores = iforest(TreeData, NumTree, NumSub)
% TreeData = Data;
% for turm_num = 1:2
rounds = 1; % rounds of repeat

% parameters for iForest
% NumTree = 100; % number of isolation trees
% NumSub = 256; % subsample size
% NumSub = 1000; % subsample size
NumDim = size(TreeData, 2); % do not perform dimension sampling 
 
auc = zeros(rounds, 1);
mtime = zeros(rounds, 2);
rseed = zeros(rounds, 1);

for r = 1:rounds
%     disp(['rounds ', num2str(r), ':']);    
    rseed(r) = sum(100 * clock);
    Forest = IsolationForest(TreeData, NumTree, NumSub, NumDim, rseed(r));
    mtime(r, 1) = Forest.ElapseTime;
    [Mass, ~] = IsolationEstimation(TreeData, Forest);
    Score = - mean(Mass, 2);   
end
% Score_see = reshape(Score,row,col);
Tree.Size=NumSub;
c = 2 * (log(Tree.Size - 1) + 0.5772156649) - 2 * (Tree.Size - 1) / Tree.Size;
scores = 2.^(Score/c);
end
