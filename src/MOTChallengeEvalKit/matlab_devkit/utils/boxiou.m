function iou=boxiou(x1,y1,w1,h1,x2,y2,w2,h2)    
% compute intersection over union of two bboxes
% 

    bisect=boxIntersect(x1,x1+w1,y1+h1,y1,x2,x2+w2,y2+h2,y2);
    iou=0;
    if ~bisect, return; end
    
    bunion=boxUnion(x1,x1+w1,y1+h1,y1,x2,x2+w2,y2+h2,y2,bisect);
    
    assert(bunion>0,'something wrong with union computation');
    iou=bisect/bunion;

end