% Tracking Performance Measures as described in the paper:
%
%  Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking.
%  E. Ristani, F. Solera, R. S. Zou, R. Cucchiara and C. Tomasi.
%  ECCV 2016 Workshop on Benchmarking Multi-Target Tracking. 
%
% Ergys Ristani 
% Duke University 2016

function [measures] = IDmeasures( groundTruthMat, predictionMat, threshold, world )
% Input: 
%    groundTruthMat   - frame, ID, left, top, width, height, worldX, worldY
%    predictionMat    - frame, ID, left, top, width, height, worldX, worldY
%    threshold        - Ground plane distance (1m) or intersection_over_union 
%    world            - boolean paramenter determining if the evaluation is
%                       done in the world ground plane or in the image plane

% Convert input trajectories from .top format to cell arrays. Each cell has 
% data for one identity.
idsPred = unique(predictionMat(:,2));
idsGT = unique(groundTruthMat(:,2));
ground_truth = cell(length(idsGT),1);
prediction = cell(length(idsPred),1);
for i = 1:length(idsGT)
   ground_truth{i} = groundTruthMat(groundTruthMat(:,2) == idsGT(i),:);
end
for i = 1:length(idsPred)
   prediction{i} = predictionMat(predictionMat(:,2) == idsPred(i),:);
end

% Initialize cost matrix blocks
cost = zeros(length(prediction) + length(ground_truth));
cost( length(ground_truth) + 1:end, 1:length(prediction)) = inf;
cost( 1:length(ground_truth), length(prediction) + 1 : end) = inf;

fp = zeros(size(cost));
fn = zeros(size(cost));

% Compute cost block
[costBlock, fpBlock, fnBlock] = costBlockMex(ground_truth, prediction, threshold, world);
% for i = 1:length(ground_truth)
%     for j = 1:length(prediction)
%         [cost(i,j), fp(i,j), fn(i,j)] = costFunction(ground_truth{i}, prediction{j}, threshold, world);
%     end
% end
cost(1:size(costBlock,1),1:size(costBlock,2)) = costBlock;
fp(1:size(costBlock,1),1:size(costBlock,2)) = fpBlock;
fn(1:size(costBlock,1),1:size(costBlock,2)) = fnBlock;

% Compute FP block
for i = 1:length(prediction)
    cost(i+length(ground_truth),i) = size(prediction{i},1);
    fp(i+length(ground_truth),i)   = size(prediction{i},1);
end

% Compute FN block
for i = 1:length(ground_truth)
    cost(i,i+length(prediction)) = size(ground_truth{i},1);
    fn(i,i+length(prediction))   = size(ground_truth{i},1);
    
end

% Solve truth-to-result identity matching 
[optimalMatch, totalCost] = MinCostMatching(cost);
for i = 1:size(optimalMatch,1)
   assignment(i) = find(optimalMatch(i,:)); 
end

% For visualization
% solutionMatrix = zeros(size(cost));
% for i = 1:length(assignment), solutionMatrix(i,assignment(i)) = 1; end

numGT = sum(cellfun(@(x) size(x,1), ground_truth));
numPRED = sum(cellfun(@(x) size(x,1), prediction));

% Count assignment errors
IDFP = 0;
IDFN = 0;
for i = 1:length(assignment)
    IDFP = IDFP + fp(i,assignment(i));
    IDFN = IDFN + fn(i,assignment(i));
end 

IDTP = numGT - IDFN;
% Sanity check
assert(IDTP == numPRED - IDFP);

IDPrecision = IDTP / (IDTP + IDFP);
if numPRED == 0, IDPrecision = 0; end
IDRecall = IDTP / (IDTP + IDFN);
IDF1 = 2*IDTP/(numGT + numPRED);

measures.IDP = IDPrecision * 100;
measures.IDR = IDRecall * 100;
measures.IDF1 = IDF1 * 100;
measures.n_gt = numGT;
measures.n_tr = numPRED;
measures.IDTP = IDTP;
measures.IDFP = IDFP;
measures.IDFN = IDFN;


