function metsBenchmark = evaluateBenchmark( allMets, world )


% Aggregate scores from all sequences
MT = 0; PT = 0; ML = 0; FRA = 0;
falsepositives = 0; missed = 0; idswitches = 0;
Fgt = 0; distsum = 0; Ngt = 0; sumg = 0;
Nc = 0; 
numGT = 0; numPRED = 0; IDTP = 0; IDFP = 0; IDFN = 0;

for ind = 1:length(allMets)
    if isempty(allMets(ind).m)
        fprintf('\n\nResults missing for sequence #%d\n', ind)
        continue; 
    end
    numGT = numGT + allMets(ind).IDmeasures.numGT;
    numPRED = numPRED + allMets(ind).IDmeasures.numPRED;
    IDTP = IDTP + allMets(ind).IDmeasures.IDTP;
    IDFN = IDFN + allMets(ind).IDmeasures.IDFN;
    IDFP = IDFP + allMets(ind).IDmeasures.IDFP;

    MT = MT + allMets(ind).additionalInfo.MT;
    PT = PT + allMets(ind).additionalInfo.PT;
    ML = ML + allMets(ind).additionalInfo.ML;
    FRA = FRA + allMets(ind).additionalInfo.FRA;
    Fgt = Fgt + allMets(ind).additionalInfo.Fgt;
    Ngt = Ngt + allMets(ind).additionalInfo.Ngt;
    Nc = Nc + sum(allMets(ind).additionalInfo.c);
    sumg = sumg + sum(allMets(ind).additionalInfo.g);
    falsepositives = falsepositives + sum(allMets(ind).additionalInfo.fp);
    missed = missed + sum(allMets(ind).additionalInfo.m);
    idswitches = idswitches + sum(allMets(ind).additionalInfo.mme);
    dists = allMets(ind).additionalInfo.d;
    td = allMets(ind).additionalInfo.td;
    distsum = distsum + sum(sum(dists));
end

IDPrecision = IDTP / (IDTP + IDFP);
IDRecall = IDTP / (IDTP + IDFN);
IDF1 = 2*IDTP/(numGT + numPRED);
if numPRED==0, IDPrecision = 0; end
IDP = IDPrecision * 100;
IDR = IDRecall * 100;
IDF1 = IDF1 * 100;


FAR = falsepositives / Fgt;
MOTP = (1-distsum/Nc) * 100; 
if world, MOTP = MOTP / td; end
if isnan(MOTP), MOTP = 0; end
MOTAL=(1-(missed+falsepositives+log10(idswitches+1))/sumg)*100;
MOTA=(1-(missed+falsepositives+idswitches)/sumg)*100;
recall=Nc/sumg*100;
precision=Nc/(falsepositives+Nc)*100;

metsBenchmark = [IDF1, IDP, IDR, recall, precision, FAR, Ngt, MT, PT, ML, falsepositives, missed, idswitches, FRA, MOTA, MOTP, MOTAL];

end

