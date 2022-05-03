function resFileClean = preprocessResult(resFile, seqName, dataDir, force, minvis)
% reads submitted (raw) MOT16 result from .txt
% and removes all boxes that are associated with ambiguous annotations
% such as sitting people, cyclists or mannequins.
% Also removes partially occluded boxes if minvis>0

% resFile='/home/amilan/research/projects/bmtt-dev/code/bae/res/0001/ADL-Rundle-6.txt';
% seqName = 'MOT16-09';
assert(cleanRequired(seqName),'preproccessing should only be done for MOT16/17 and MOT20')

if nargin<4, force=1; end
if nargin<4, force=1; end
if nargin<5, minvis=0; end

fprintf('Preprocessing (cleaning) %s...\n',seqName);


% if file does not exist, do nothing
if ~exist(resFile,'file')
    fprintf('Results file does not exist\n');
    resFileClean = []; 
    return;
end

[p,f,e]=fileparts(resFile);
cleanDir = [p,filesep,'clean'];
if ~exist(cleanDir, 'dir'), mkdir(cleanDir); end
resFileClean = [cleanDir,filesep,f,e];

% if clean file already exists and no need to redo, skip
if ~force && exist(resFileClean, 'file')
    fprintf('skipping...\n');
    return;
end

% if file empty, just copy it
tf = dir(resFile);
if tf.bytes == 0
    fprintf('Results file empty\n');
    copyfile(resFile,resFileClean);
    return;
end

if nargin<3, dataDir = getDataDir; end

% [seqName, seqFolder, imgFolder, imgExt, F, dirImages] ...
%     = getSeqInfo(seq, dataDir);

[seqName, seqFolder, imgFolder, frameRate, F, imWidth, imHeight, imgExt] ...
    = getSeqInfoFromFile(seqName, dataDir);

% read in result
resRaw = dlmread(resFile);



%%

% read ground truth
gtFolder = [dataDir,filesep,'gt',filesep];
gtFile = [gtFolder,'gt.txt'];
gtRaw = dlmread(gtFile);

% make sure we have MOT16 ground truth (= 9 columns)
assert(size(gtRaw,2)==9, 'unknown GT format')

% define which classes should be ignored
if ~isempty(strfind(seqName,'MOT20'))
    distractors = {'person_on_vhcl','static_person','distractor','reflection', 'non_mot_vhcl'};

else 
    distractors = {'person_on_vhcl','static_person','distractor','reflection'};
end

keepBoxes = true(size(resRaw,1),1);

showVis = 0;

td=0.5; % overlap threshold
for t=1:F
    if ~mod(t,100), fprintf('.'); end
    
    % find all result boxes in this frame
    resInFrame = find(resRaw(:,1)==t); N = length(resInFrame);
    resInFrame = reshape(resInFrame,1,N);
    
    % find all GT boxes in frame
    GTInFrame = find(gtRaw(:,1)==t); Ngt = length(GTInFrame);
    GTInFrame = reshape(GTInFrame,1,Ngt);
    
    % compute all overlaps for current frame
    allisects=zeros(Ngt,N);
    g=0;
    for gg=GTInFrame
        g=g+1; r=0;
        bxgt=gtRaw(gg,3); bygt=gtRaw(gg,4); bwgt=gtRaw(gg,5); bhgt=gtRaw(gg,6);
        for rr=resInFrame
            r=r+1;
            bxres=resRaw(rr,3); byres=resRaw(rr,4); bwres=resRaw(rr,5); bhres=resRaw(rr,6);
            
            if bxgt+bwgt<bxres, continue; end % ignore if no horizontal overlap
            if bxgt>bxres+bwres, continue; end
            
            if bygt+bhgt<byres, continue; end % ignore if no vertical overlap
            if bygt>byres+bhres, continue; end
            
            allisects(g,r)=boxiou(bxgt,bygt,bwgt,bhgt,bxres,byres,bwres,bhres);
        end
    end
%     t
    
    tmpai=allisects;
    tmpai=1-tmpai;
    tmpai(tmpai>td)=Inf;
    [Mtch,Cst]=Hungarian(tmpai);
    [mGT,mRes]=find(Mtch);
%     pause
    nMtch = length(mGT);
    % go through all matches
    for m=1:nMtch        
        g=GTInFrame(mGT(m)); % gt box 
        gtClassID = gtRaw(g,8);
        gtClassString = classIDToString(gtClassID);
        
        % if we encounter a distractor, mark to remove box
        if ismember(gtClassString, distractors)
            r = resInFrame(mRes(m)); % result box
            keepBoxes(r) = false;
            
            if showVis
                bxgt=gtRaw(g,3); bygt=gtRaw(g,4); bwgt=gtRaw(g,5); bhgt=gtRaw(g,6); idgt=gtRaw(g,2);
                bxres=resRaw(r,3); byres=resRaw(r,4); bwres=resRaw(r,5); bhres=resRaw(r,6); idres=resRaw(r,2);

                clf
                im = imread(fullfile(dataDir,seqName,'img1',sprintf('%06d.jpg',t)));
                imshow(im); hold on
                text(50,50,sprintf('%d',t),'color','w')
                
                % show GT box
%                 text(bxgt,bygt-20,sprintf('%d',idgt),'color','w')
                classString = insertEscapeChars(classIDToString(gtClassID));
                text(bxgt+50,bygt-20,sprintf('%s',classString),'color','w')     
                rectangle('Position',[bxgt,bygt,bwgt,bhgt],'EdgeColor','w');
                
                % show Res box
%                 text(bxres,byres-20,sprintf('%d',idres),'color','y')
                rectangle('Position',[bxres,byres,bwres,bhres],'EdgeColor','y');
                
                pause(.01)
            end
        end
        
        % if we encounter a partially occluded box, mark to remove
        if gtRaw(g,9)<minvis
            r = resInFrame(mRes(m)); % result box
            keepBoxes(r) = false;
            
            if showVis
                bxgt=gtRaw(g,3); bygt=gtRaw(g,4); bwgt=gtRaw(g,5); bhgt=gtRaw(g,6); idgt=gtRaw(g,2);
                bxres=resRaw(r,3); byres=resRaw(r,4); bwres=resRaw(r,5); bhres=resRaw(r,6); idres=resRaw(r,2);

                clf
                im = imread(fullfile(dataDir,seqName,'img1',sprintf('%06d.jpg',t)));
                imshow(im); hold on
                text(50,50,sprintf('%d',t),'color','w')
                
                % show GT box
                text(bxgt+50,bygt-20,sprintf('vis %.1f',gtRaw(g,9)*100),'color','w')     
                rectangle('Position',[bxgt,bygt,bwgt,bhgt],'EdgeColor','w');
                
                % show Res box
%                 text(bxres,byres-20,sprintf('%d',idres),'color','y')
                rectangle('Position',[bxres,byres,bwres,bhres],'EdgeColor','y');
                
                pause(.01)
                pause
            end
        end        
        
    end
end

%%
fprintf('\nRemoving %d boxes from %s solution...\n',[numel(find(~keepBoxes)), seqName]);
resNew = resRaw;
resNew=resNew(keepBoxes,:);

%% write new file into new dir (clean)
dlmwrite(resFileClean, resNew);




% end
