function cl=cleanRequired(seqFolder)

cl =~isempty(strfind(seqFolder,'MOT20')) || ~isempty(strfind(seqFolder,'MOT16')) || ~isempty(strfind(seqFolder,'MOT17')) || ~isempty(strfind(seqFolder,'HOTA'));
