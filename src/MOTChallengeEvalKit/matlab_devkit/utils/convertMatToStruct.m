function stInfo=convertMatToStruct(allData, gtMaxFrame)

numCols=size(allData,2);
numLines=size(allData,1);

% quickly check format
assert(numCols>=7,'FORMAT ERROR: Each line must have at least 6 values');
% assert(all(allData(:,1)>0),'FORMAT ERROR: Frame numbers must be positive.');
% assert(all(allData(:,2)>0),'FORMAT ERROR: IDs must be positive.');

IDMap = unique(allData(:,2))';
nIDs = length(IDMap);
% fprintf('%d unique targets IDs discovered\n',nIDs);
% pause

% do we have bbox coordinates?
imCoord=0;
if any(any(allData(:,3:6)~= -1))
    imCoord=1;
end

% do we have world coordinates?
worldCoord=0;

% ... we do if they exist and are not -1 (default)
if size(allData,2)>=9
    if any(any(allData(:,[8 9])~= -1))
        worldCoord=1; 
    end
end

% at least one representation must be defined
assert(imCoord || worldCoord, ...
    'FORMAT ERROR: Neither bounding boxes nor world coorsinates defined.');
    
% go through all lines
for l=numLines:-1:1
    lineData=allData(l,:);
    
    fr = lineData(1);   % frame number
    id = lineData(2);   % target id

    % map id to 1..nIDs
    id = find(IDMap==id);
    
    %%%% sanity checks
    % ignore non-positive frames and IDs
    if fr<1,
        continue;
    end

    % bounding box    
    stInfo.W(fr,id) = lineData(5);
    stInfo.H(fr,id) = lineData(6);
    stInfo.Xi(fr,id) = lineData(3) + stInfo.W(fr,id)/2;
    stInfo.Yi(fr,id) = lineData(4) + stInfo.H(fr,id);
    
    % values should not be exactly 0
    if ~stInfo.W(fr,id)
        stInfo.W(fr,id) = stInfo.W(fr,id) + 0.0001;
    end
    if ~stInfo.H(fr,id)
        stInfo.H(fr,id) = stInfo.H(fr,id) + 0.0001;
    end
    if ~stInfo.Xi(fr,id)
        stInfo.Xi(fr,id) = stInfo.Xi(fr,id) + 0.0001;
    end
    if ~stInfo.Yi(fr,id)
        stInfo.Yi(fr,id) = stInfo.Yi(fr,id) + 0.0001;
    end
    % consider 3D coordinates
    if worldCoord
        stInfo.Xgp(fr,id) = lineData(7);
        stInfo.Ygp(fr,id) = lineData(8);
        
        % position should not be exactly 0
        if ~stInfo.Xgp(fr,id)
            stInfo.Xgp(fr,id)=stInfo.Xgp(fr,id)+0.0001;
        end
        if ~stInfo.Ygp(fr,id)
            stInfo.Ygp(fr,id)=stInfo.Ygp(fr,id)+0.0001;
        end
    end
end

% append empty frames?
if gtMaxFrame ~= -1
   
    Fgt=gtMaxFrame;
    F=size(stInfo.W,1);
    % if stateInfo shorter, pad with zeros
    if F<Fgt
        missingFrames = F+1:Fgt;
        stInfo.Xi(missingFrames,:)=0;
        stInfo.Yi(missingFrames,:)=0;
        stInfo.W(missingFrames,:)=0;
        stInfo.H(missingFrames,:)=0;
        if worldCoord
            stInfo.Xgp(missingFrames,:)=0;
            stInfo.Ygp(missingFrames,:)=0;
        end
    end
end

% set X,Y
stInfo.X=stInfo.Xi;stInfo.Y=stInfo.Yi;
if ~imCoord 
    stInfo.X=stInfo.Xgp;stInfo.Y=stInfo.Ygp; 
    
    % reset image coordinates to -1
    te = find(stInfo.X(:));
    stInfo.W(te)=-1;stInfo.H(te)=-1;
    stInfo.Xi(te)=-1;stInfo.Yi(te)=-1;
end


% remove empty target IDs
nzc=~~sum(abs(stInfo.Xi),1);

if isfield(stInfo,'X')
    stInfo.X=stInfo.X(:,nzc);
    stInfo.Y=stInfo.Y(:,nzc);
end

% nzc
% stInfo.Xgp'
if isfield(stInfo,'Xgp')
    stInfo.Xgp=stInfo.Xgp(:,nzc);
    stInfo.Ygp=stInfo.Ygp(:,nzc); 
end

if isfield(stInfo,'Xi')
    stInfo.Xi=stInfo.Xi(:,nzc);
    stInfo.Yi=stInfo.Yi(:,nzc); 
    stInfo.W=stInfo.W(:,nzc);
    stInfo.H=stInfo.H(:,nzc);
end