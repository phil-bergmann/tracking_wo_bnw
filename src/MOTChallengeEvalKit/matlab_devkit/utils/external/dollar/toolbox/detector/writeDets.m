function writeDets(bbx,outfile)
% write out Dollar's detections to our format

F=length(bbx);

alldets=zeros(0,10);
for t=1:F
    ndets=size(bbx{t},1);    
    dets=[repmat(t,ndets,1) repmat(-1,ndets,1) bbx{t} repmat(-1,ndets,1) repmat(-1,ndets,1) repmat(-1,ndets,1)];
    alldets=[alldets; dets];    
end
dlmwrite(outfile,alldets);

end
