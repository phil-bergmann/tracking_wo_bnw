function [ metsMultiCam ] = evaluateMultiCam( gtMat, resMat, threshold, world )
%EVALUATEMULTICAM Summary of this function goes here
%   Detailed explanation goes here
% Prepare data for overall evaluation
gtMatAll = [];
resMatAll = [];
countF = 0;
for k = 1:length(gtMat)
    newF =  max(max(gtMat{k}(:,1)),max(resMat{k}(:,1)));
    if isempty(newF), newF = 0; end
    gtMat{k}(:,1) = gtMat{k}(:,1) + countF;
    resMat{k}(:,1) = resMat{k}(:,1) + countF;
    countF = countF + newF;
    gtMatAll = [gtMatAll; gtMat{k}];
    resMatAll = [resMatAll; resMat{k}];
end

metsID = IDmeasures(gtMatAll, resMatAll, threshold, world);
metsMultiCam = [metsID.IDF1, metsID.IDP, metsID.IDR];





