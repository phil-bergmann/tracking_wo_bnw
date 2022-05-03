function [cost, fp, fn] = costFunction( tr1, tr2, THRESHOLD, world )

frames_gt   = tr1(:,2);
frames_pred = tr2(:,2);

% If trajectories don't overlap in time then return cost or inf
overlapTest = (frames_gt(1) >= frames_pred(1) && frames_gt(1) < frames_pred(end)) || ...
              (frames_gt(end) >= frames_pred(1) && frames_gt(end) <= frames_pred(end)) || ...
              (frames_pred(1) >= frames_gt(1) && frames_pred(1) <= frames_gt(end)) || ...
              (frames_pred(end) >= frames_gt(1) && frames_pred(end) <= frames_gt(end));

if ~overlapTest
    fp = length(frames_pred);
    fn = length(frames_gt);
    cost = fp + fn;
    return;
end

[isfoundGT, posGT]   = ismember_mex(frames_gt, frames_pred);
[isfoundPred, posPred] = ismember_mex(frames_pred, frames_gt);

% Use points at infinity when no match exists
columns = [7,8];
if ~world
    columns = [3 4 5 6];
end

% Ground truth data and the corresponding data from the prediction
pointsGT = tr1(isfoundGT, columns);
pointsGTPred = tr2(posGT(isfoundGT),columns);

% Prediction data and the corresponding data from ground truth
pointsPred = tr2(isfoundPred, columns);
pointsPredGT = tr1(posPred(isfoundPred),columns);

unmatchedGT = sum(isfoundGT==0);
unmatchedPred = sum(isfoundPred==0);

distanceGTvsPred = distanceFunction(pointsGT, pointsGTPred, world);
distancePredvsGT = distanceFunction(pointsPred, pointsPredGT, world);

if world % Euclidean distance test
    fn = unmatchedGT + sum( distanceGTvsPred > THRESHOLD );
    fp = unmatchedPred + sum( distancePredvsGT > THRESHOLD );
else % IntersectionOverUnion test
    fn = unmatchedGT + sum( distanceGTvsPred < THRESHOLD );
    fp = unmatchedPred + sum( distancePredvsGT < THRESHOLD );
end
cost = fp + fn;

% Sanity check
tp1 = length(frames_gt) - fn;
tp2 = length(frames_pred) - fp;

assert(tp1 == tp2, 'Something is wrong in the input. Make sure there are no duplicate frames...');


end

function distance = distanceFunction(point1, point2, world)

if world
    
    % Euclidean distance
    distance = sqrt(sum(abs(point1 - point2).^2,2));
    
else
    
    % Intersection_over_union
    box1 = point1;
    box2 = point2;
    
    area1 = box1(:,3) .* box1(:,4);
    area2 = box2(:,3) .* box2(:,4);
    
    l1 = box1(:,1); r1 = box1(:,1) + box1(:,3); t1 = box1(:,2); b1 = box1(:,2) + box1(:,4);
    l2 = box2(:,1); r2 = box2(:,1) + box2(:,3); t2 = box2(:,2); b2 = box2(:,2) + box2(:,4);
    
    x_overlap = max(0, min(r1,r2) - max(l1,l2));
    y_overlap = max(0, min(b1,b2) - max(t1,t2));
    intersectionArea = x_overlap .* y_overlap;
    unionArea = area1 + area2 - intersectionArea;
    iou = intersectionArea ./ unionArea;
    
    distance = iou;
    
end



end

