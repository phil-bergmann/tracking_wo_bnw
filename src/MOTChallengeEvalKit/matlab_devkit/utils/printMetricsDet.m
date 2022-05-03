function printMetricsDet(metrics, metricsInfo, dispHeader,dispMetrics,padChar)
% print metrics
% 
% ...
%

% default names
if nargin==1
    
metricsInfo.names.long = {'Recall','Precision','False Alarm Rate', ...
    'Ground Truth', 'True Positives', 'False Positives', 'False Negatives', 'MODA','MODP'};

metricsInfo.names.short = {'Rcll','Prcn','FAR', ...
    'GT', 'TP', 'FP', 'FN', 'MODA','MODP'};

    metricsInfo.widths.long = [6 9 16 15 15 15 15 5 5];
    metricsInfo.widths.short = [5 5 5 6 6 6 6  5 5];

metricsInfo.format.long = {'.1f','.1f','.2f', ...
    'i','i','i','i', '.1f','.1f'};

metricsInfo.format.short=metricsInfo.format.long;
end

namesToDisplay=metricsInfo.names.long;
widthsToDisplay=metricsInfo.widths.long;
formatToDisplay=metricsInfo.format.long;

namesToDisplay=metricsInfo.names.short;
widthsToDisplay=metricsInfo.widths.short;
formatToDisplay=metricsInfo.format.short;

if nargin<3, dispHeader=1; end
if nargin<4
    dispMetrics=1:length(metrics);
end
if nargin<5
    padChar={' ','|',' ',' ',' ',' ','|',' ',' '};
end

if dispHeader
    for m=dispMetrics
        printString=sprintf('fprintf(''%%%is%s'',char(namesToDisplay(m)))',widthsToDisplay(m),char(padChar(m)));
        eval(printString)
    end
    fprintf('\n');
end

for m=dispMetrics
    printString=sprintf('fprintf(''%%%i%s%s'',metrics(m))',widthsToDisplay(m),char(formatToDisplay(m)),char(padChar(m)));
    eval(printString)
end

% if standard, new line
if nargin<4
    fprintf('\n');
end