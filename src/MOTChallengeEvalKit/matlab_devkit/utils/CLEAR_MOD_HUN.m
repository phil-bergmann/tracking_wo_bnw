function [metrics, metricsInfo, additionalInfo]=CLEAR_MOD_HUN(gt,det,options)
% compute CLEAR Detection metrics according to
% PERFORMANCE EVALUATION PROTOCOL FOR FACE, PERSON AND
%  VEHICLE DETECTION & TRACKING IN VIDEO ANALYSIS AND
%  CONTENT EXTRACTION (VACE-II)
%  CLEAR – CLASSIFICATION OF EVENTS, ACTIVITIES AND RELATIONSHIPS
%  Submitted to Advanced Research and Development Activity

% metrics contains the following
% [1]   recall	- recall = percentage of detected targets
% [2]   precision	- precision = percentage of correctly detected targets
% [3]   FAR		- number of false alarms per frame
% [4]   GT      - number of GT boxes
% [5]   truepositivies- number of true positives (TP)
% [6]   falsepositives- number of false positives (FP)
% [7]   missed        - number of missed targets (FN)
% [8]	MODA          - N-MODA
% [9]	MODP          - N-MODP
% 
% 

%
% 
% (C) Anton Milan, 2017
%
% Input
% MOTChallenge CSV format
% gt - ground truth
% det - detection result


% default options: 2D
if nargin<3
    options.eval3d=0;   % only bounding box overlap
    options.td=.5;      % threshold 50%
end

if ~isfield(options,'td')
    if options.eval3d
        options.td=1000;
        error('not implemented');
    else
        options.td=0.5;
    end
end

td=options.td;

% if X,Y not existent, assume 2D
%  if ~isfield(gtInfo,'X'), gtInfo.X=gtInfo.Xi; end
%  if ~isfield(gtInfo,'Y'), gtInfo.Y=gtInfo.Yi; end
%  if ~isfield(stateInfo,'X'), stateInfo.X=stateInfo.Xi; end
%  if ~isfield(stateInfo,'Y'), stateInfo.Y=stateInfo.Yi; end
%  
%  gtInd=~~gtInfo.X;
%  stInd=~~stateInfo.X;
%  
%  [Fgt, Ngt]=size(gtInfo.X);
%  [F, N]=size(stateInfo.X);
Fgt=max(gt(:,1)); Ngt=max(gt(:,2));
F=max(det(:,1)); N=max(det(:,2));

%  
%  % if stateInfo shorter, pad with zeros
%  if F<Fgt
%      missingFrames = F+1:Fgt;
%      stateInfo.Xi(missingFrames,:)=0;
%      stateInfo.Yi(missingFrames,:)=0;
%      stateInfo.W(missingFrames,:)=0;
%      stateInfo.H(missingFrames,:)=0;
%  end

% [1]   recall	- recall = percentage of detected targets
% [2]   precision	- precision = percentage of correctly detected targets
% [3]   FAR		- number of false alarms per frame
% [4]   falsepositives- number of false positives (FP)
% [5]   missed        - number of missed targets (FN)
% [6]	MODA          - N-MODA
% [7]	MODP		  - N-MODP
% [8]	MODP-t
% [9]	MODA-t


metricsInfo.names.long = {'Recall','Precision','False Alarm Rate', ...
    'Ground Truth', 'True Positives', 'False Positives', 'False Negatives', 'MODA','MODP'};

metricsInfo.names.short = {'Rcll','Prcn','FAR', ...
    'GT', 'TP', 'FP', 'FN', 'MODA','MODP'};

    metricsInfo.widths.long = [6 9 16 15 15 15 15 5 5];
    metricsInfo.widths.short = [5 5 5 6 6 6 6 5 5];

metricsInfo.format.long = {'.1f','.1f','.2f', ...
    'i','i','i','i', '.1f','.1f'};

metricsInfo.format.short=metricsInfo.format.long;


metrics=zeros(1,7);
metrics(5)=size(gt,1);  % False Negatives (missed)

additionalInfo=[];
% nothing to be done, if state is empty
if ~N, return; end


% mapping
M=zeros(F,Ngt);

%  mme=zeros(1,F); % ID Switchtes (mismatches)
c=zeros(1,F);   % matches found
fp=zeros(1,F);  % false positives
m=zeros(1,F);   % misses = false negatives
g=zeros(1,F);
d=zeros(F,Ngt);  % all distances;
ious=Inf*ones(F,Ngt);  % all overlaps

%  matched=@matched2d;
%  if options.eval3d, matched=@matched3d; end

alltracked=zeros(F,Ngt);
allfalsepos=zeros(F,N);

for t=1:F

%     if ~mod(t,1000), fprintf('.'); end % print every 1000th frame
    
    % mapping for current frame
%      if t>1
%          mappings=find(M(t-1,:));
%          for map=mappings
%              if gtInd(t,map) && stInd(t,M(t-1,map)) && matched(gtInfo,stateInfo,t,map,M(t-1,map),td)
%                  M(t,map)=M(t-1,map);
%              end
%          end
%      end
    
%     GTsInFrame=find(~M(t,:) & gtInd(t,:));
%     DetsInFrame=setdiff(find(stInd(t,:)),M(t,:));
    
%      stIndInT = find(stInd(t,:));
%      GTsInFrame=find(~M(t,:) & gtInd(t,:));
%      DetsInFrame=stIndInT;
%      if ~isempty(stIndInT) && any(M(t,:))
%          DetsInFrame=setdiff(stIndInT,M(t,:));
%      end
    
    GTsInFrame = find(gt(:,1)==t);
    DetsInFrame = find(det(:,1)==t);
    

    
    % reshape to ensure horizontal vector in empty case
    DetsInFrame=reshape(DetsInFrame,1,length(DetsInFrame));
    GTsInFrame=reshape(GTsInFrame,1,length(GTsInFrame));
    
    Ngtt=length(GTsInFrame);
    Nt = length(DetsInFrame);
    g(t)=Ngtt;
	
    if options.eval3d
        alldist=Inf*ones(Ngtt, Nt);
    
        mindist=0;
        for o=1:Ngtt
            GT=[gt(GTsInFrame(o),8), gt(GTsInFrame(o),9)];
            for e=1:Nt
                E=[det(DetsInFrame,8), det(DetsInFrame(e),9)];
                alldist(ocnt,ecnt)=norm(GT-E);
            end
        end
            
        
        tmpai=alldist;
        tmpai(tmpai>td)=Inf;
        [Mtch,~]=Hungarian(tmpai);
        [u,v]=find(Mtch);
        
        for mmm=1:length(u)
            M(t,u(mmm))=v(mmm);
        end

    
    else
        if ~isempty(GTsInFrame) && ~isempty(DetsInFrame)
            allisects=zeros(Ngtt,Nt);        maxisect=Inf;

            for o=1:Ngtt
				GT=gt(GTsInFrame(o),[3:6]);
                for e=1:Nt
					E=det(DetsInFrame(e),[3:6]);
                    allisects(o,e)=boxiou(GT(1),GT(2),GT(3),GT(4),E(1),E(2),E(3),E(4));
                end
            end


            tmpai=allisects;
            tmpai=1-tmpai;
            tmpai(tmpai>td)=Inf;

            % do Hungarian matching only if there is anything to match
            if numel(find(~isinf(tmpai)))>0
                [Mtch,~]=Hungarian(tmpai);
                [u,v]=find(Mtch);

%                 M=M;
                for mmm=1:length(u)
                    M(t,u(mmm))=v(mmm);
                end
            end
        end

    end
    
    curdetected=find(M(t,:)); % which GTs are detected?
    
    
    alldetections=1:Nt;
    
%     mappedDets=intersect(M(t,find(M(t,:))),alldetections);
%     falsepositives=setdiff(alldetections,mappedDets);
    mappedDets = [];
    if ~isempty(alldetections) && any(M(t,curdetected))
        mappedDets=intersect(M(t,curdetected),alldetections);
    end
    
    falsepositives=alldetections;
    if ~isempty(falsepositives) && ~isempty(mappedDets) && any(mappedDets)
        falsepositives=setdiff(alldetections,mappedDets);
    end
    
    alltracked(t,:)=M(t,:);
%     allfalsepos(t,1:length(falsepositives))=falsepositives;
    allfalsepos(t,falsepositives)=falsepositives;
    
    c(t)=numel(curdetected);
    for ct=curdetected
        eid=M(t,ct);
        if options.eval3d
            d(t,ct)=norm([gt(GTsInFrame(ct),8),  gt(GTsInFrame(ct),9)] - ...
                [det(DetsInFrame(eid),8),  det(DetsInFrame(eid),9)]);
        else
			gtLeft = gt(GTsInFrame(ct), 3);
			gtTop = gt(GTsInFrame(ct), 4);
			gtWidth = gt(GTsInFrame(ct), 5);
			gtHeight = gt(GTsInFrame(ct), 6);
			
			stLeft = det(DetsInFrame(eid), 3);
			stTop = det(DetsInFrame(eid), 4);
			stWidth = det(DetsInFrame(eid), 5);
			stHeight = det(DetsInFrame(eid), 6);
            ious(t,ct)=boxiou(gtLeft,gtTop,gtWidth,gtHeight,stLeft,stTop,stWidth,stHeight);
        end
    end
    
    
    fp(t)=Nt-c(t);
    m(t)=g(t)-c(t);
    
    
end    

missed=sum(m);
falsepositives=sum(fp);
truepositives=sum(c);
%  idswitches=sum(mme);

if options.eval3d
    MODP=(1-sum(sum(d))/sum(c)/td) * 100; % avg distance to [0,100]
else
    MODP=sum(ious(ious>=td & ious<Inf))/sum(c) * 100; % avg ol
end
if isnan(MODP), MODP=0; end % force to 0 if no matches found

MODA=(1-((sum(m)+sum(fp))/sum(g)))*100;
recall=sum(c)/sum(g)*100;
precision=sum(c)/(sum(fp)+sum(c))*100;
FAR=sum(fp)/Fgt;
GT=sum(g);
 
metrics=[recall, precision, FAR, GT, truepositives, falsepositives, missed, MODA, MODP];

%additionalInfo.alltracked=alltracked;
%additionalInfo.allfalsepos=allfalsepos;

additionalInfo.fn = sum(m);
additionalInfo.fp = sum(fp);
%ground truth objects
additionalInfo.n_gt = sum(g);
additionalInfo.tp = sum(c);
additionalInfo.total_num_frames = Fgt;
additionalInfo.n_gt_trajectories = Ngt;
additionalInfo.ious = ious;
additionalInfo.td = td;

end