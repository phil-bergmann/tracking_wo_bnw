function classString = classIDToString(classID)

labels = getClassLabels;

if classID<1 || classID>length(labels)
    classString='unknown';
else
    classString = char(labels{classID});
end

end