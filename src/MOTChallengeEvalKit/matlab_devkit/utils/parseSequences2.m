function allseq = parseSequences2(seqmapFile)
% parse sequence map
% seqmapFile - a file containing the sequence names
%    to be processed. First line is ignored. e.g.
%
% --------------
% name
% TUD-Stadtmitte
% TUD-Campus
% --------------
%
% returns a cell array with all the sequence names


assert(exist(seqmapFile,'file')>0,'seqmap file %s does not exist',seqmapFile);
fid = fopen(seqmapFile);
allseq = textscan(fid,'%s','HeaderLines',1);
fclose(fid);
allseq=allseq{1}';
    
    
end
