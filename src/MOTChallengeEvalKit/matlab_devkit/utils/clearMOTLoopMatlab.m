function [mme, c, fp, m, g, d, ious, alltracked, allfalsepos, M] = clearMOTLoopMatlab(gtInfo, stateInfo, options)

gtInd=~~gtInfo.W;
stInd=~~stateInfo.W;

[Fgt, Ngt]=size(gtInfo.X);
[F, N]=size(stateInfo.X);
F = Fgt;

% mapping
M=zeros(F,Ngt);
M=sparse(M);

mme=zeros(1,F); % ID Switchtes (mismatches)
c=zeros(1,F);   % matches found
fp=zeros(1,F);  % false positives
m=zeros(1,F);   % misses = false negatives
g=zeros(1,F);
d=zeros(F,Ngt);  % all distances;
ious=Inf*ones(F,Ngt);  % all overlaps

matched=@matched2d;
if options.eval3d, matched=@matched3d; end

td = options.td;


alltracked=zeros(F,Ngt);
allfalsepos=zeros(F,N);

for t=1:F
    g(t)=numel(find(gtInd(t,:)));
    
    % mapping for current frame
    if t>1
        mappings=find(M(t-1,:));
        for map=mappings
            if gtInd(t,map) && stInd(t,M(t-1,map)) && matched(gtInfo,stateInfo,t,map,M(t-1,map),td)
                M(t,map)=M(t-1,map);
                if options.VERBOSE
                    fprintf('%d: preserve %d\n', t, map);
                end
            end
        end
    end
    
    GTsNotMapped=find(~M(t,:) & gtInd(t,:));
    EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));

    % reshape to ensure horizontal vector in empty case
	EsNotMapped=reshape(EsNotMapped,1,length(EsNotMapped));
	GTsNotMapped=reshape(GTsNotMapped,1,length(GTsNotMapped));

    if options.eval3d
        alldist=Inf*ones(Ngt,N);
    
        uid = 0; % unique identifier to break ties
        for o=GTsNotMapped
            GT=[gtInfo.Xgp(t,o) gtInfo.Ygp(t,o)];
            for e=EsNotMapped
                E=[stateInfo.Xgp(t,e) stateInfo.Ygp(t,e)];
                alldist(o,e)=norm(GT-E);
                if alldist(o,e) > td
                    alldist(o,e) = 1e8;
                else
                    alldist(o,e) = alldist(o,e) + 1e-9 * uid;
                    uid = uid + 1;
                end
            end
        end
        
        tmpai = alldist;
        tmpai_compact = tmpai(GTsNotMapped, EsNotMapped);
        [Mtch,Cst]=MinCostMatching(tmpai_compact);
        [u,v]=find(Mtch);
        [u, iu] = sort(u);
        v = v(iu);
        u = GTsNotMapped(u);
        v = EsNotMapped(v);
        
        for mmm=1:length(u)
            if tmpai(u(mmm),v(mmm)) == 1e8
                continue;
            end
            M(t,u(mmm))=v(mmm);
            if options.VERBOSE
                fprintf('%d: map %d with %d\n', t, u(mmm), v(mmm));
            end
        end
        
        
%         while mindist < td && numel(GTsNotMapped)>0 && numel(EsNotMapped)>0
%             for o=GTsNotMapped
%                 GT=[gtInfo.Xgp(t,o) gtInfo.Ygp(t,o)];
%                 for e=EsNotMapped
%                     E=[stateInfo.Xgp(t,e) stateInfo.Ygp(t,e)];
%                     alldist(o,e)=norm(GT-E);
%                 end
%             end
%             
%             [mindist cind]=min(alldist(:));
% 
%             if mindist <= td
%                 [u v]=ind2sub(size(alldist),cind);
%                 M(t,u)=v;
%                 alldist(:,v)=Inf;
%                 GTsNotMapped=find(~M(t,:) & gtInd(t,:));
%                 EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));
%             end
%         end
    
    else
        allisects=zeros(Ngt,N);        
        tmpai = Inf * ones(size(allisects));
        uid = 0;
        for o=GTsNotMapped
            GT=[gtInfo.X(t,o)-gtInfo.W(t,o)/2 ...
                gtInfo.Y(t,o)-gtInfo.H(t,o) ...
                gtInfo.W(t,o) gtInfo.H(t,o) ];
            for e=EsNotMapped
                E=[stateInfo.Xi(t,e)-stateInfo.W(t,e)/2 ...
                    stateInfo.Yi(t,e)-stateInfo.H(t,e) ...
                    stateInfo.W(t,e) stateInfo.H(t,e) ];
                allisects(o,e)= boxiou(GT(1),GT(2),GT(3),GT(4),E(1),E(2),E(3),E(4));
                tmpai(o,e) = 1 - allisects(o,e);
                if tmpai(o,e) > td
                    tmpai(o,e) = 1e8; 
                else
                    tmpai(o,e) = tmpai(o,e) + 1e-9 * uid;
                    uid = uid + 1;
                end
            end
        end
        
        if options.VERBOSE
            fprintf('%d: UnmappedGTs: ', t);
            for i = 1:length(GTsNotMapped)
                fprintf('%d, ', GTsNotMapped(i));
            end
            fprintf('\n%d: UnmappedEs: ', t);
            for i = 1:length(EsNotMapped)
                fprintf('%d, ', EsNotMapped(i));
            end
            fprintf('\n');
        end
            
        tmpai_compact = tmpai(GTsNotMapped, EsNotMapped);
        [Mtch,Cst]=MinCostMatching(tmpai_compact);
        [u,v]=find(Mtch);
        [u, iu] = sort(u);
        v = v(iu);
        u = GTsNotMapped(u);
        v = EsNotMapped(v);

        for mmm=1:length(u)
            if tmpai(u(mmm),v(mmm)) == 1e8
                continue;
            end
            M(t,u(mmm))=v(mmm);
            if options.VERBOSE
                fprintf('%d: map %d with %d\n', t, u(mmm), v(mmm));
            end
        end
        
%         GTsNotMapped=find(~M(t,:) & gtInd(t,:));
%         EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));
            
%         while maxisect > td && numel(GTsNotMapped)>0 && numel(EsNotMapped)>0
% 
%             for o=GTsNotMapped
%                 GT=[gtInfo.X(t,o)-gtInfo.W(t,o)/2 ...
%                     gtInfo.Y(t,o)-gtInfo.H(t,o) ...
%                     gtInfo.W(t,o) gtInfo.H(t,o) ];
%                 for e=EsNotMapped
%                     E=[stateInfo.Xi(t,e)-stateInfo.W(t,e)/2 ...
%                         stateInfo.Yi(t,e)-stateInfo.H(t,e) ...
%                         stateInfo.W(t,e) stateInfo.H(t,e) ];
%                     allisects(o,e)=boxiou(GT(1),GT(2),GT(3),GT(4),E(1),E(2),E(3),E(4));
%                 end
%             end
%             
%             [maxisect, cind]=max(allisects(:));
% 
%             if maxisect >= td
%                 [u, v]=ind2sub(size(allisects),cind);
%                 M(t,u)=v;
%                 allisects(:,v)=0;
%                 GTsNotMapped=find(~M(t,:) & gtInd(t,:));
%                 EsNotMapped=setdiff(find(stInd(t,:)),M(t,:));
%             end
% 
%         end
    end
    
    curtracked=find(M(t,:));
    
    
    alltrackers=find(stInd(t,:));
    mappedtrackers=intersect(M(t,find(M(t,:))),alltrackers);
    falsepositives=setdiff(alltrackers,mappedtrackers);
    
    alltracked(t,:)=M(t,:);
%     allfalsepos(t,1:length(falsepositives))=falsepositives;
    allfalsepos(t,falsepositives)=falsepositives;
    
    %%  mismatch errors
    if t>1
        for ct=curtracked
            lastnotempty=find(M(1:t-1,ct),1,'last');
            if gtInd(t-1,ct) && ~isempty(lastnotempty) && M(t,ct)~=M(lastnotempty,ct)
                mme(t)=mme(t)+1;
            end
        end
    end
    
    c(t)=numel(curtracked);
    for ct=curtracked
        eid=M(t,ct);
        if options.eval3d
            d(t,ct)=norm([gtInfo.Xgp(t,ct) gtInfo.Ygp(t,ct)] - ...
                [stateInfo.Xgp(t,eid) stateInfo.Ygp(t,eid)]);
        else
            gtLeft=gtInfo.X(t,ct)-gtInfo.W(t,ct)/2;
            gtTop=gtInfo.Y(t,ct)-gtInfo.H(t,ct);
            gtWidth=gtInfo.W(t,ct);    gtHeight=gtInfo.H(t,ct);
            
            stLeft=stateInfo.Xi(t,eid)-stateInfo.W(t,eid)/2;
            stTop=stateInfo.Yi(t,eid)-stateInfo.H(t,eid);
            stWidth=stateInfo.W(t,eid);    stHeight=stateInfo.H(t,eid);
            ious(t,ct)=boxiou(gtLeft,gtTop,gtWidth,gtHeight,stLeft,stTop,stWidth,stHeight);
        end
    end
    
    
    fp(t)=numel(find(stInd(t,:)))-c(t);
    m(t)=g(t)-c(t);
    
    
end     
%
if ~options.eval3d
    d = max(0,1-ious);
end
end

function ret=matched2d(gtInfo,stateInfo,t,map,mID,td)
    gtLeft=gtInfo.X(t,map)-gtInfo.W(t,map)/2;
    gtTop=gtInfo.Y(t,map)-gtInfo.H(t,map);
    gtWidth=gtInfo.W(t,map);    gtHeight=gtInfo.H(t,map);
    
    stLeft=stateInfo.Xi(t,mID)-stateInfo.W(t,mID)/2;
    stTop=stateInfo.Yi(t,mID)-stateInfo.H(t,mID);
    stWidth=stateInfo.W(t,mID);    stHeight=stateInfo.H(t,mID);
    
    ret = boxiou(gtLeft,gtTop,gtWidth,gtHeight,stLeft,stTop,stWidth,stHeight) >= td;    
end


function ret=matched3d(gtInfo,stateInfo,t,map,mID,td)
    Xgt=gtInfo.Xgp(t,map); Ygt=gtInfo.Ygp(t,map);
    X=stateInfo.Xgp(t,mID); Y=stateInfo.Ygp(t,mID);
    ret=norm([Xgt Ygt]-[X Y])<=td;

end