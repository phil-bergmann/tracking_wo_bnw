function union=boxUnion(bboxleft1, bboxright1, bboxbottom1, bboxup1, bboxleft2, bboxright2, bboxbottom2, bboxup2,isect)
% Compute union of two bounding boxes

a1=bboxright1-bboxleft1;
b1=bboxbottom1-bboxup1;
a2=bboxright2-bboxleft2;
b2=bboxbottom2-bboxup2;
union=a1*b1+a2*b2;
if nargin>8
    bisect=isect;
else
    bisect=boxIntersect(bboxleft1, bboxright1, bboxbottom1, bboxup1, bboxleft2, bboxright2, bboxbottom2, bboxup2);
end
union=union-bisect;


end
