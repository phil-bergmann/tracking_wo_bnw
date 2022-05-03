function detectPeople(sequence,thr)
% run P. Dollar's pedestrian detector.
% sequence can either be a string (for one sequence only)
% or a cell array, for multiple sequences.
% If sequence is empty, run on all available data
%
% thr is the cutoff threshold (unused)
%
% @author: Anton Milan

addpath(genpath('..')); % dollar toolbox
addpath(genpath('../../../../scripts')) % tools

datadir=getDataDir();



if ~nargin, sequence=[]; end
allseq = parseSequences(sequence,datadir);

load('models/AcfInriaDetector.mat')

detScale=1;
detScale=0.6;

blowUp=1;
fprintf('RESCALING DETECTOR: %f\n',detScale);
detector = acfModify(detector,'rescale',detScale);

parpool(4);
parfor sn=1:length(allseq)
    seq=allseq(sn);
    seqName=char(seq);
	fprintf('Processing sequence %s\n',seqName)
    seqFolder= [datadir,seqName,filesep];
    if ~exist(seqFolder,'dir')
        fprintf('WARNING: ''%s does not exist''\n',seqFolder);
        continue;
    end
    
    imgExt=getImgExt(seqFolder);
    imgFolders = dir(fullfile(seqFolder,filesep,'img*'));
	if length(imgFolders)<1
		fprintf('WARNING: No image folders found in %s\n', seqFolder)
	end
    for iF=1:length(imgFolders)
        imgFolder = [seqFolder,imgFolders(iF).name,filesep];
        
        
        imgMask=[imgFolder,'*',imgExt];
        %         imgMask=[imgFolder,'*.jpg'];
        dirImages = dir(imgMask);
        
        if isempty(dirImages)
            fprintf('WARNING: No images found in ''%s''\n',imgFolder);
            continue;
        end
        
        F=length(dirImages);
%         F=10;
        filecells=cell(1,F);
        for t=1:F
            filecells{t} = [imgFolder,dirImages(t).name];
        end
        
        fprintf('Detecting %s (%d frames)\n',seqName,F);
        
        % detection's folder and file
        detFolder = [datadir,seqName,filesep,'det',filesep];
        if ~exist(detFolder,'dir'), mkdir(detFolder); end
        detFile = [detFolder,'det-acf.txt'];

	delete(detFile);
	delete([detFolder,'det-preclean.txt']);
	delete([detFolder,'det-orig.txt']);
        
        % detect all images
        %         detector.opts.pNms.type='none';
        bbx = acfDetect(filecells,detector);
        
        % write out
        writeDets(bbx,detFile);
        
        % create preview (first image)
        try
            clf;
            detfr=1;
            img=dirImages(detfr).name;
            frame = imread([imgFolder,   img]);
            [imH,imW,imC]=size(frame);
            imagesc(frame);
            
            set(gca,'XTick',[]);  set(gca,'YTick',[]);
            set(gca,'position',[0 0 1 1],'units','normalized')
            % set(gcf,'Units','pixels','Position',[4 4 400 300]);
            
            
            hold on
            for b=1:size(bbx{detfr},1)
                rectangle('Position',bbx{detfr}(b,1:4),'linewidth',2,'EdgeColor','w');
                text(bbx{detfr}(b,1)+1,bbx{detfr}(b,2)+1,sprintf('%.2f',bbx{detfr}(b,5)),'color','w', ...
                    'VerticalAlign','top','FontWeight','bold');
            end
            detpic=[detFolder,sprintf('%06d-acf.jpg',detfr)];
%             export_fig(detpic,'-a1','-native','painters')

            saveas(gcf,detpic);
            if exist(detpic,'file')
                imd=imread(detpic);
                imd=imresize(imd,[imH,imW]);
                imwrite(imd,detpic);
            end
        catch err
            fprintf('ERROR: %s',err.identifier);
        end
    end
end

delete(gcp);
