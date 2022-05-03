function isect=boxIntersect(bboxleft1, bboxright1, bboxbottom1, bboxup1, bboxleft2, bboxright2, bboxbottom2, bboxup2)
% Compute intersection area of two bouding boxes A,B
% A=[bboxleft1 bboxbottom1 abs(bboxright1-bboxleft1) abs(bboxbottom1-bboxup1)];
% B=[bboxleft2 bboxbottom2 abs(bboxright2-bboxleft2) abs(bboxbottom2-bboxup2)];
% 
% isect=rectint(A,B);        
isect=0;

hor= max(0,min(bboxright1,bboxright2) - max(bboxleft1,bboxleft2));

if ~hor, return; end
ver= max(0,min(bboxbottom1,bboxbottom2) - max(bboxup1,bboxup2));
if ~ver, return; end

isect = hor*ver;

end