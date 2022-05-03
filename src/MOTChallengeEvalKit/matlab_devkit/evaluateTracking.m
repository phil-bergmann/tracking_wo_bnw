function [sequenceName, mets, metsID, additionalInfo, results]=evaluateTracking(sequenceName, resFilename, gtFilename, gtDataDir, benchmark)
% seqmap, resDir, gtDataDir, benchmark
% Input:
% - seqmap
% Sequence map (e.g. `c2-train.txt` contains a list of all sequences to be
% evaluated in a single run. These files are inside the ./seqmaps folder.
%
% - resDir
% The folder containing the tracking results. Each one should be saved in a
% separate .txt file with the name of the respective sequence (see ./res/data)
%
% - gtDataDir
% The folder containing the ground truth files.
%
% - benchmark
% The name of the benchmark, e.g. 'MOT15', 'MOT16', 'MOT17', 'DukeMTMCT'
%
% Output:
% - allMets
% Scores for each sequence
%
% - metsBenchmark
% Aggregate score over all sequences
%
% - metsMultiCam
% Scores for multi-camera evaluation


addpath(genpath('.'));
addpath(genpath('/srv/motchallenge/MOTChallenge/motchallenge-devkit'));
warning off;
% Benchmark specific properties
world = 0;
threshold = 0.5;
multicam = 0;


gtMat = [];
resMat = [];

% Evaluate sequences individually
allMets = {};
sequences = {};
metrics = {};
metsBenchmark = [];
metsMultiCam = [];

gtdata = dlmread(gtFilename);

gtdata(gtdata(:,7)==0,:) = [];     % ignore 0-marked GT
gtdata(gtdata(:,1)<1,:) = [];      % ignore negative frames


if strcmp(benchmark, 'MOT16') || strcmp(benchmark, 'MOT17')  || strcmp(benchmark, 'MOT20') % ignore non-pedestrians
    gtdata(gtdata(:,8)~=1,:) = [];
end

if strcmp(benchmark, 'MOT15_3D')
    gtdata(:,[7 8]) = gtdata(:,[8 9]); % shift world coordinates
end
[~, ~, ic] = unique(gtdata(:,2)); % normalize IDs
gtdata(:,2) = ic;
%gtMat{ind} = gtdata;


% Parse result

% MOTX data format

if strcmp(benchmark, 'MOT16') || strcmp(benchmark, 'MOT17') || strcmp(benchmark, 'MOT20')
    resFilename = preprocessResult(resFilename, sequenceName, gtDataDir);
end

% Skip evaluation if output is missing
if ~exist(resFilename,'file')
    error('Invalid submission. Result for sequence %s not available!\n',sequenceName);
end

% Read result file
if exist(resFilename,'file')
    s = dir(resFilename);
    if s.bytes ~= 0
        resdata = dlmread(resFilename);
    else
        resdata = zeros(0,9);
    end
else
    error('Invalid submission. Result file for sequence %s is missing or invalid\n', resFilename);
end
resdata(resdata(:,1)<1,:) = [];      % ignore negative frames
if strcmp(benchmark, 'MOT15_3D')
    resdata(:,[7 8]) = resdata(:,[8 9]);  % shift world coordinates
end
resdata(resdata(:,1) > max(gtdata(:,1)),:) = []; % clip result to gtMaxFrame
%resMat{ind} = resdata;



% Sanity check
frameIdPairs = resdata(:,1:2);
[u,I,~] = unique(frameIdPairs, 'rows', 'first');
hasDuplicates = size(u,1) < size(frameIdPairs,1);
if hasDuplicates
    ixDupRows = setdiff(1:size(frameIdPairs,1), I);
    dupFrameIdExample = frameIdPairs(ixDupRows(1),:);
    rows = find(ismember(frameIdPairs, dupFrameIdExample, 'rows'));

    errorMessage = sprintf('Invalid submission: Found duplicate ID/Frame pairs in sequence %s.\nInstance:\n', sequenceName);
    errorMessage = [errorMessage, sprintf('%10.2f', resdata(rows(1),:)), sprintf('\n')];
    errorMessage = [errorMessage, sprintf('%10.2f', resdata(rows(2),:)), sprintf('\n')];
    assert(~hasDuplicates, errorMessage);
end

% Evaluate sequence
[metsCLEAR, mInf, additionalInfo] = CLEAR_MOT_HUN(gtdata, resdata, threshold, world);

metsID = IDmeasures(gtdata, resdata, threshold, world);
mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];

keySet = mInf.names.short;
valueSet = mInf.names.long;




results = struct();

metsIDnames = fieldnames(metsID);
for k=1:numel(metsIDnames)
	results = setfield(results, metsIDnames{k},metsID.(metsIDnames{k}));
end

addInfonames = fieldnames(additionalInfo);

for k=1:length(addInfonames)
results = setfield(results, addInfonames{k},additionalInfo.(addInfonames{k}));
end

end