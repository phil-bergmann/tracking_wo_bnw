function [seqName, seqFolder, imgFolder, frameRate, F, imWidth, imHeight, imgExt] ...
    = getSeqInfoFromFile(seq, dataDir)
% construct variables with relevant information about the sequence 'seq'
% 

    seqName=char(seq);
    seqFolder= [dataDir,filesep];
    

    seqInfoFile = [dataDir, filesep,'seqinfo.ini'];
    ini = IniConfig();
    ini.ReadFile(seqInfoFile);
    
    imgFolder = ini.GetValues('Sequence','imDir');
    frameRate = ini.GetValues('Sequence','frameRate');
    F=ini.GetValues('Sequence','seqLength');
    imWidth=ini.GetValues('Sequence','imWidth');
    imHeight=ini.GetValues('Sequence','imHeight');
    imgExt = ini.GetValues('Sequence','imExt');
    
end
