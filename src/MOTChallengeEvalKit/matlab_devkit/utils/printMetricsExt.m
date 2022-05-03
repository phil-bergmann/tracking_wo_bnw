function printMetricsExt(metrics, metricsInfo, dispHeader,dispMetrics,padChar)
% print metrics
% 
% ...
%

% default names
if nargin==1
    metricsInfo.names.long = {'IDF1', 'IDP', 'IDR',    'Recall','Precision','False Alarm Rate', ...
        'GT Tracks','Mostly Tracked','Partially Tracked','Mostly Lost', ...
        'False Positives', 'False Negatives', 'ID Switches', 'Fragmentations', ...
        'MOTA','MOTP', 'MOTA Log'};

    metricsInfo.names.short = {'IDF1', 'IDP', 'IDR',    'Rcll','Prcn','FAR', ...
        'GT','MT','PT','ML', ...
        'FP', 'FN', 'IDs', 'FM', ...
        'MOTA','MOTP', 'MOTAL'};

    metricsInfo.widths.long = [5 4 4   6 9 16 9 14 17 11 15 15 11 14 5 5 8];
    metricsInfo.widths.short = [5 4 4  5 5 5 4 4 4 4 6 6 5 5 5 5 5];

    metricsInfo.format.long = {'.1f','.1f','.1f', ...
        '.1f','.1f','.2f', ...
        'i','i','i','i', ...
        'i','i','i','i', ...
        '.1f','.1f','.1f'};

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
    padChar={' ',' ','|',' ',' ','| ','',' ',' ','|','',' ',' ','| ',' ',' ',' ',' '};
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