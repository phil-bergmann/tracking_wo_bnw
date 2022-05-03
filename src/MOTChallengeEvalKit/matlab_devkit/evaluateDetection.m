function results = evaluateDetection(seqName, resFile, gtFilename, gtDataDir, benchmark)
% seqmap,resDir,dataDir, benchmark
%% evaluate detections using P. Dollar's script

this_files_folder = fileparts(which(mfilename));
addpath(genpath(this_files_folder));
% addpath(genpath('.'));

fprintf('Challenge: %s\n',benchmark)
fprintf('Sequence: %s\n', seqName);


gtInfo=[];
gtInfo.X=[];

cls = [2,7,8,12]; %% ambiguous classes
minvis = 0.5;
ref=0:.025:1;
ref=0:.1:1;
showEachRef=1;


% Find out the length of each sequence
% and concatenate ground truth
gtInfoSingle=[];
gtAll={};
detInfoSingle=[];
detAll={};
seqCnt=0;
allFrCnt=0;
evalMethod=1;
gtAllMatrix=zeros(0,6);
detAllMatrix=zeros(0,7);
% read gt data
gtRaw = dlmread(gtFilename);


% if something (a result) is missing, we cannot evaluate this tracker
[seqName, seqFolder, imgFolder, imgExt, F, dirImages] ...
    = getSeqInfoFromFile(seqName, gtDataDir);
if ~exist(resFile,'file')
    fprintf('WARNING: result for %s not available: %s\n',seqName, resFile);
    evalMethod=0;
end
% set visibility threshold to 25% for MOT 20
if ~isempty(strfind(benchmark,'MOT20'))
    minvis = 0.25;
    fprintf('Min Vis changed: \t %s', minvis) ;
end

% if MOT16, MOT17 or MOT20 preprocess (clean)
if cleanRequired(benchmark)
    resFile = preprocessResult(resFile, seqName, gtDataDir, 1, minvis);
end

detRaw=dlmread(resFile);

%
gtOne= {};
detOne = {};
for t=1:F
    allFrCnt=allFrCnt+1;

    % keep pedestrians only and vis >= minvis
    exgt=find(gtRaw(:,1)==t & gtRaw(:,8)==1 & gtRaw(:,9)>=minvis);
    gtAll{allFrCnt}=[gtRaw(exgt,3:6) zeros(length(exgt),1)];
    gtOne{t}=[gtRaw(exgt,3:6) zeros(length(exgt),1)];

    ng = length(exgt);
    oneFrame=[allFrCnt*ones(ng,1), (1:ng)', gtRaw(exgt,3:6)]; % set IDs to 1..ng
    gtAllMatrix=[gtAllMatrix; oneFrame];

    exdet=find(detRaw(:,1)==t);
    bbox=detRaw(exdet,3:7);
    detAll{allFrCnt}=bbox;
    detOne{t}=bbox;

    ng = length(exdet);
    oneFrame=[allFrCnt*ones(ng,1), (1:ng)', detRaw(exdet,3:7)]; % set IDs to 1..ng
    detAllMatrix=[detAllMatrix; oneFrame];
end


allFgt= F;
gtInfoSingle.gt=gtOne;
gtInfoSingle.gtMat=gtRaw(find(gtRaw(:,8)==1 & gtRaw(:,9)>=minvis),1:6);
detInfoSingle.det = detOne;
detInfoSingle.detMat = detRaw;


detResults=[];




try

    % iterate over each sequence



        fprintf('\t... %s\n',seqName);


        gt0=gtInfoSingle.gt;
        dt0=detInfoSingle.det;
        [gt,dt]=bbGt('evalRes',gt0,dt0);
        [rc,pr,scores,refprcn, tp,  np] = bbGt('compRoc',gt,dt,0,ref);

        AP = mean(refprcn);

		detResults.np=np;
		detResults.scores=scores;
		detResults.tp=tp;
        detResults.rc=rc;
        detResults.pr=pr;
        detResults.ref=refprcn;
        detResults.AP=AP;
        detResults.name=seqName;



        gtRawPed = gtInfoSingle.gtMat;
        detRawPed = detInfoSingle.detMat;
        [detMets, detMetsInfo, detMetsAddInfo]=CLEAR_MOD_HUN(gtRawPed,detRawPed);
        detResults.detMets = detMets;
        detResults.detMetsInfo = detMetsInfo;
        detResults.detMetsAddInfo = detMetsAddInfo;
		results = struct();

		detMetsAddInfoNames = fieldnames(detMetsAddInfo);
		for k=1:numel(detMetsAddInfoNames)
			results = setfield(results, detMetsAddInfoNames{k},detMetsAddInfo.(detMetsAddInfoNames{k}));
		end
		results = setfield(results, "refprcn",refprcn);
		results = setfield(results,"scores",scores);
		results = setfield(results, "tp_list", tp);

        refprstr = '';
        for r=1:length(refprcn)
            refprstr=[refprstr,sprintf('%.4f',refprcn(r))];
            if r<length(refprcn), refprstr=[refprstr,',']; end
        end



        refprcn = detResults.ref;
        AP = detResults.AP;
        detMets = detResults.detMets;

        fprintf('\t... %s\n',seqName);
        fprintf('Recall:    ')
        for r=1:showEachRef:length(ref)
            fprintf('%6.3f',ref(r));
        end
        fprintf('\n')
        fprintf('Precision: ')
        for r=1:showEachRef:length(ref)
            fprintf('%6.3f',refprcn(r));
        end
        fprintf('\n');
        fprintf('Average Precision: %.4f\n',AP);
        printMetrics(detMets);

        fprintf('\n');



        fprintf('\n\n');




catch err
    fprintf('WARNING: Cannot be evaluated: %s\n',err.message);
    getReport(err)
end

