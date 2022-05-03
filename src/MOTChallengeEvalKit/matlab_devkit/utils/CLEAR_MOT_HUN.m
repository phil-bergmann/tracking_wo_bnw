function [metrics, metricsInfo, additionalInfo] = CLEAR_MOT_HUN(gtMat, resMat, threshold, world)
% compute CLEAR MOT and other metrics
%
% metrics contains the following
% [1]   recall	- percentage of detected targets
% [2]   precision	- percentage of correctly detected targets
% [3]   FAR		- number of false alarms per frame
% [4]   GT        - number of ground truth trajectories
% [5-7] MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories
% [8]   falsepositives- number of false positives (FP)
% [9]   missed        - number of missed targets (FN)
% [10]  idswitches	- number of id switches     (IDs)
% [11]  FRA       - number of fragmentations
% [12]  MOTA	- Multi-object tracking accuracy in [0,100]
% [13]  MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
% [14]  MOTAL	- Multi-object tracking accuracy in [0,100] with log10(idswitches)
%
% 
% (C) Anton Milan, 2012-2014

metricsInfo.names.long = ["Recall","Precision","False Alarm Rate", ...
    "GT Tracks","Mostly Tracked","Partially Tracked","Mostly Lost", ...
    "False Positives", "False Negatives", "ID Switches", "Fragmentations", ...
    "MOTA","MOTP", "MOTA Log"];

metricsInfo.names.short = ["Rcll","Prcn","FAR", ...
    "GT","MT","PT","ML", ...
    "FP", "FN", "IDs", "FM", ...
    "MOTA","MOTP", "MOTAL"];

metricsInfo.widths.long = [6 9 16 9 14 17 11 13 15 15 11 14 5 5 8];
metricsInfo.widths.short = [5 5 5 3 3 3 3 2 4 4 3 3 5 5 5];

metricsInfo.format.long = {'.1f','.1f','.2f', ...
    'i','i','i','i', ...
    'i','i','i','i','i', ...
    '.1f','.1f','.1f'};

metricsInfo.format.short=metricsInfo.format.long;
additionalInfo=[];

% Normalize IDs
[~, ~, ic] = unique(gtMat(:,2)); gtMat(:,2) = ic;
[~, ~, ic2] = unique(resMat(:,2)); resMat(:,2) = ic2;

% Evaluate
VERBOSE = false;
[mme, c, fp, m, g, d, alltracked, allfalsepos] = clearMOTMex(gtMat, resMat, threshold, world, VERBOSE);
Fgt = max(gtMat(:,1)); % Assumes first gt frame is 1
Ngt = length(unique(gtMat(:,2)));
F = max(resMat(:,1));
missed=sum(m);
falsepositives=sum(fp);
idswitches=sum(mme);



MOTP=(1-sum(sum(d))/sum(c)) * 100; % avg distance to [0,100]
if world, MOTP = MOTP / threshold; end
if isnan(MOTP), MOTP=0; end % force to 0 if no matches found

MOTAL=(1-((sum(m)+sum(fp)+log10(sum(mme)+1))/sum(g)))*100;
MOTA=(1-((sum(m)+sum(fp)+(sum(mme)))/sum(g)))*100;
recall=sum(c)/sum(g)*100;
precision=sum(c)/(sum(fp)+sum(c))*100;
if isnan(precision), precision=0; end % force to 0 if no matches found
FAR=sum(fp)/Fgt;
 

%% MT PT ML
MTstatsa=zeros(1,Ngt);
for i=1:Ngt
    gtframes = gtMat(gtMat(:,2)==i,1);
    gttotallength=numel(gtframes);
    trlengtha=numel(find(alltracked(gtframes,i)>0));
    if trlengtha/gttotallength < 0.2
        MTstatsa(i)=3;
    elseif F>=find(gtMat(gtMat(:,2)==i,1),1,'last') && trlengtha/gttotallength <= 0.8
        MTstatsa(i)=2;
    elseif trlengtha/gttotallength >= 0.8
        MTstatsa(i)=1;
    end
end
% MTstatsa
MT=numel(find(MTstatsa==1));
PT=numel(find(MTstatsa==2));
ML=numel(find(MTstatsa==3));

%% fragments
fr=zeros(1,Ngt);
for i=1:Ngt
    b=alltracked(find(alltracked(:,i),1,'first'):find(alltracked(:,i),1,'last'),i);
    b(~~b)=1;
    fr(i)=numel(find(diff(b)==-1));
end
FRA=sum(fr);

% assert(Ngt==MT+PT+ML,'Hmm... Not all tracks classified correctly.');
metrics=[recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL];

%additionalInfo.alltracked=alltracked;
%additionalInfo.allfalsepos=allfalsepos;
additionalInfo.fn = missed;
additionalInfo.fp = sum(fp);
additionalInfo.id_switches = idswitches;

additionalInfo.tp = sum(c);
additionalInfo.total_num_frames = Fgt;
additionalInfo.n_gt_trajectories = Ngt;
additionalInfo.total_cost = sum(sum(d));


additionalInfo.MT = MT;
additionalInfo.PT = PT;
additionalInfo.ML = ML;
additionalInfo.FM = FRA;
additionalInfo.td = threshold;


end 



