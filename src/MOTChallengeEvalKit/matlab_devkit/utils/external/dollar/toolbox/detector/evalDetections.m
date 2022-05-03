%%

addpath(genpath('..'));
addpath(genpath('../../../../scripts'));

resDir = getResDir;
dataDir = getDataDir;

challenge = 2; % 2D MOT 2015
% challenge = 5; % MOT16

% eval all
challenge = 2; % 2D MOT 2015
challenge = 5; % MOT16

detectorName = 'acf';
detectorName = 'dpm';
detectorName = 'fasterrcnndet-th-0';
% detectorName = '';

sequences=[];
% allSeq={'TUD-Stadtmitte'};
% sequences='PETS09-S2L1';
% sequences={'TUD-Stadtmitte','PETS09-S2L1'};
allSeq = parseSequences([0 challenge],dataDir);
% allSeq = {'MOT16-01'}

    dt0=[];
    gt0=[];
% for each sequence
allFrCnt=0;
for seq=allSeq    

    [seqName, seqFolder, imgFolder, imgExt, F, dirImages] ...
        = getSeqInfo(seq, dataDir);

    
%     detFolder = [dataDir,seqName,filesep,'det',filesep];
%     detFile = [detFolder,'det.txt'];
    [detFolder, detFile]=getDetInfo(seqName,dataDir,detectorName);
    
%     if ~exist(detFile,'file'), continue; end

    gtFolder= [dataDir,seqName,filesep,'gt',filesep];
    gtFile = [gtFolder,'gt.txt'];
    
    detRaw=dlmread(detFile);
    gtRaw=dlmread(gtFile);
    

        
    for t=1:F
        allFrCnt=allFrCnt+1;
        exdet=find(detRaw(:,1)==t);
        bbox=detRaw(exdet,3:7);
        dt0{allFrCnt}=bbox;
        
        exgt=find(gtRaw(:,1)==t & gtRaw(:,8)==1);
        gt0{allFrCnt}=[gtRaw(exgt,3:6) zeros(length(exgt),1)];
        
    end
end

allSeq = parseSequences([1 challenge],dataDir);
% allSeq = {'MOT16-02'}

    dt1=[];
    gt1=[];

% for each sequence
allFrCnt1=0;
for seq=allSeq    

    [seqName, seqFolder, imgFolder, imgExt, F, dirImages] ...
        = getSeqInfo(seq, dataDir);

    
%     detFolder = [dataDir,seqName,filesep,'det',filesep];
%     detFile = [detFolder,'det.txt'];
    [detFolder, detFile]=getDetInfo(seqName,dataDir,detectorName);
%     if ~exist(detFile,'file'), continue; end

    gtFolder= [dataDir,seqName,filesep,'gt',filesep];
    gtFile = [gtFolder,'gt.txt'];
    
    detRaw=dlmread(detFile);
    gtRaw=dlmread(gtFile);
    
        
    for t=1:F
        allFrCnt1=allFrCnt1+1;
        exdet=find(detRaw(:,1)==t);
        bbox=detRaw(exdet,3:7);
        dt1{allFrCnt1}=bbox;
        
        exgt=find(gtRaw(:,1)==t & gtRaw(:,8)==1);
        gt1{allFrCnt1}=[gtRaw(exgt,3:6) zeros(length(exgt),1)];
        
    end
end



% lims=[3.1e-3 1e1 .05 1];
lims=[0 1 0 1];
ref=10.^(-2:.25:0);
ref=0:.1:1;

%%
clf; hold on; box on;

% Training Set
[gt,dt]=bbGt('evalRes',gt0,dt0);
[fp,tp,score,miss] = bbGt('compRoc',gt,dt,0,ref);

% Test Set
[gt,dt]=bbGt('evalRes',gt1,dt1);
[fp1,tp1,score1,miss1] = bbGt('compRoc',gt,dt,0,ref);

set(gca,'FontSize',12)
plot(fp,tp,'linewidth',2);
plot(fp1,tp1,'linewidth',2,'color','r');

opX=fp(end); opY=tp(end);
plot(opX,opY,'.','MarkerSize',25);
opX1=fp1(end); opY1=tp1(end);
plot(opX1,opY1,'.','MarkerSize',25,'color','r');

%%
fs=22;
set(gca,'FontSize',fs);
text(opX+.25,opY+.1,sprintf('Rcll: %.1f %%\nPrc: %.1f %%',opX*100,opY*100),'VerticalAlignment','top','HorizontalAlignment','right','FontSize',fs-4);
text(opX1-.05,opY1-.025,sprintf('Rcll: %.1f %%\nPrc: %.1f %%',opX1*100,opY1*100),'VerticalAlignment','top','HorizontalAlignment','right','FontSize',fs-4);
xlim(lims(1:2)); ylim(lims(3:4));


xlabel('Recall'); ylabel('Precision');
title('Detector Performance');
legend('Training Set','Test Set');

% export_fig 'det-dollar.pdf' -png -pdf;
% saveas(gcf,'det-dollar.pdf');
% saveas(gcf,'det-dollar.png');
% export_fig 'det-dollar.pdf' -transparent;
pdfFileName = sprintf('det-%s.pdf',detectorName);
export_fig(pdfFileName, '-transparent');
% export_fig '../../../../../../papers/2015/arxiv-benchmark/figures/det-dollar.pdf' -transparent;



% miss=exp(mean(log(max(1e-10,1-miss)))); roc=[score fp tp];
% figure(show); plotRoc([fp tp],'logx',1,'logy',0,'xLbl','fppi',...
%   'lims',lims,'color','g','smooth',1,'fpTarget',ref);
% title(sprintf('log-average miss rate = %.2f%%',miss*100));
