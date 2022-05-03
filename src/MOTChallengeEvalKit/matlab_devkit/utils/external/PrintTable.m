classdef PrintTable < handle
% PrintTable: Class that allows table-like output spaced by tabs for multiple rows.
%
% Allows to create a table-like output for multiple values and columns.
% A spacer can be used to distinguish different columns (PrintTable.ColSep) and a flag
% (PrintTable.HasHeader) can be set to insert a line of dashes after the first row.
% The table can be printed directly to the console or be written to a file.
%
% LaTeX support:
% Additionally, LaTeX is supported via the property PrintTable.Format. The default value is
% 'plain', which means simple output as formatted string. If set to 'tex', the table is printed
% in a LaTeX table environment (PrintTable.ColSep is ignored and '& ' used automatically).
%
% Cell contents:
% The cell content values can be anything that is straightforwardly parseable. You can pass
% char array arguments directly or numeric values; even function handles and classes (handle
% subclasses) can be passed and will automatically be converted to a string representation.
%
% Custom cell content formatting:
% However, if you need to have a customized string representation, you can specifiy a cell
% array of strings as the last argument, containing custom formats to apply for each passed
% argument.
% Two conditions apply for this case: 
% # If the cell contains more than one value, there must be one argument
% for each column of the PrintTable
% # The column contents must be valid arguments for sprintf if containing a
% string, or hold a function handle taking one argument and returning a
% char array.
%
% When passing a function handle, it may also take two arguments. The
% second argument is then used to pass in the column number which is
% currently formatted.
%
% Transposing:
% An overload for the ctranspose-method of MatLab is available which easily switches rows with
% columns.
%
% Examples:
% % Simply run
% PrintTable.test_PrintTable;
% % or
% PrintTable.test_PrintTable_RowHeader_Caption;
% % or
% PrintTable.test_PrintTable_LaTeX_Export;
%
% % Or copy & paste
% t = PrintTable;
% t.addRow('123','456','789');
% t.addRow('1234567','1234567','789');
% t.addRow('1234567','12345678','789');
% t.addRow('12345678','123','789');
% % sprintf-format compatible strings can also be passed as last argument:
% % single format argument:
% t.addRow(123.456789,pi,789,{'%3.4f'});
% % custom format for each element:
% t.addRow(123.456789,pi,789,{'%3.4f','%g','format. dec.:%d'});
% t.addRow('123456789','12345678910','789');
% t.addRow('adgag',uint8(4),4.6);
% t.addRow(@(mu)mu*4+56*245647869,t,'adgag');
% t.addRow('adgag',4,4.6);
% % Call display
% t.display;
% t.HasHeader = true;
% % or simply state the variable to print
% t
%
% % Transpose the table:
% tt = t';
% tt.Caption = 'This is me, but transposed!';
% tt.print;
%
% % To use a different column separator just set e.g.
% t.ColSep = ' -@- ';
% t.display;
%
% % PrintTable with "row header" mode
% t = PrintTable('This is PrintTable-RowHeader-Caption test, created on %s',datestr(now));
% t.HasRowHeader = true;
% t.HasHeader = true;
% t.addRow('A','B','C');
% t.addRow('header-autofmt',456,789,{'%d'});
% t.addRow(1234.345,456,789,{'%2.2E','%d','%d'});
% t.addRow('header-expl-fmt',456,pi,{'%s','%d','%2.2f'});
% t.addRow('nofmt-header',456,pi,{'%d','%f'});
% t.addRow('1234567','12345','789');
% t.addRow('foo','bar',datestr(clock));
% t.display;
%
% % Latex output
% t.Format = 'tex';
% t.Caption = 'My PrintTable in LaTeX!';
% t.print;
%
% % Printing the table:
% % You can also print the table to a file. Any MatLab file handle can be used (any first
% % argument for fprintf). Run the above example, then type
% fid = fopen('mytable.txt','w');
% % [..do some printing here ..]
% t.print(fid);
% % [..do some printing here ..]
% fclose(fid);
%
% % Saving the table to a file:
% % Plaintext
% t.saveToFile('mytable_plain.txt');
% % LaTeX
% t.saveToFile('mytable_tex.tex');
% % Tight PDF
% t.saveToFile('mytable_tightpdf.pdf');
% % Whole page PDF
% t.TightPDF = false;
% t.saveToFile('mytable_tightpdf.pdf');
%
%
% @note Of course, any editor might have its own setting regarding tab
% spacing. As the default in MatLab and e.g. KWrite is four characters,
% this is what is used here. Change the TabCharLen constant to fit to your
% platform/editor/requirements.
% 
% See also: fprintf sprintf
%
% @author Daniel Wirtz @date 2011-11-17
%
% Those links were helpful for programming the PDF export:
% - http://tex.stackexchange.com/questions/2917/resize-paper-to-mbox
% - http://tex.stackexchange.com/questions/22173
% - http://www.weinelt.de/latex/
%
% @change{0,7,dw,2013-08-10} The format cell argument can now also hold
% function handles for advanced formatting. 
%
% @change{0,7,dw,2013-04-09} Added some table cell spacing by default (`\LaTeX` 'arraystretch')
%
% @change{0,7,dw,2013-04-05}
% - Automatically stripping newline characters from any strings
% - Added some more verbose output about table creation to `\LaTeX` output
%
% @new{0,7,dw,2013-02-19} Added a new method "append" to join data from a second table into the
% current one. Specific columns may be given.
%
% @change{0,7,dw,2013-02-19}
% - Bugfix: Accidentally also wrapped in $$ for plain text output by default.
%
% @change{0,7,dw,2013-02-15}
% - Bugfix for empty cell string
% - LaTeX alignment for first column is "r" if HasRowHeader is used
%
% @new{0,7,dw,2013-02-14}
% - Added a new property "TexMathModeDetection" that automatically detects (via str2num and \
% containment) if a cell content can be interpreted as a latex math mode value and wraps it in
% dollar characters when LaTeX output is set. This is switched on by default.
% - New property "StripInsertedTabChars" which causes automatic stripping of tab characters
% passed as arguments or created by sprintf formats or callbacks. This is switched on by
% default.
% - Added a short comment describing the applied settings to `\LaTeX` output.
% - Added a removeRow method.
% - Smaller improvements for LaTex export & added test/demo cases
%
% @change{0,7,dw,2012-12-11} Improved automatic setting of TabCharLen default values.
%
% @new{0,6,dw,2012-09-19} Added printing support for function handles and improved output for
% numerical values
%
% @new{0,6,dw,2012-07-16} 
% - Added an overload for the "ctranspose" method of MatLab; now easy switching of rows and
% columns is possible. Keeping the HasHeader flag if set.
% - Fixed a problem with LaTeX export when specifying only a filename without "./" in the path
% - Made the TabCharLen a public property as MatLab behaves strange with respect to tabs for
% some reason.
%
% @change{0,6,dw,2012-06-11} 
% - Added a new property NumRows that returns the number of rows (excluding
% the header if set).
%- Made the output a bit nicer and supporting logical values now
%
% @change{0,6,dw,2012-05-04}
% - Added a property PrintTable.HasRowHeader that allows to use a single
% format specification for all row entries but the first one. Added a test
% for that.
% - A caption with sprintf-compatible arguments can be passed directly to
% the PrintTable constructor now.
% - Added support for pdf export (requires pdflatex on PATH)
% - The saveToFile method either opens a save dialog to pick a file or
% takes a filename.
% - New property PrintTable.TightPDF that determines if the pdf file
% generated for a table should be cropped to the actual table size or
% inserted into a standard article document. Having a tight pdf removes the
% caption of the table.
%
% @change{0,6,dw,2011-12-14} Added support for arrays of PrintTable instances in display, print
% and saveToFile methods.
%
% @new{0,6,dw,2011-12-01}
% - Added support for LaTeX output
% - New properties PrintTable.Format and PrintTable.Caption
% - Optional caption can be added to the Table
% - Some improvements and fixed display for some special cases
% - New PrintTable.clear method
% - Updated the documentation and test case
%
% @new{0,6,dw,2011-11-17} Added this class.
%
% This class has originally been developed as part of the framework
% KerMor - Model Order Reduction using Kernels:
% - \c Homepage http://www.agh.ians.uni-stuttgart.de/research/software/kermor.html
% - \c Documentation http://www.agh.ians.uni-stuttgart.de/documentation/kermor/
%
% Copyright (c) 2011, Daniel Wirtz
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification, are
% permitted only in compliance with the BSD license, see
% http://www.opensource.org/licenses/bsd-license.php
%
% @todo replace fprintf by sprintf and build string that can be returned by this.print and if
% no output argument is collected directly print it.

    properties 
	    % Equivalent length of a tab character in single-space characters
        %
        % @default PrintTable.DefaultTabCharLen @type integer
        %
        % See also: PrintTable.getDefaultTabCharLen
        TabCharLen = PrintTable.getDefaultTabCharLen;
	   
        % A char sequence to separate the different columns.
        %
        % @default ' | ' @type char
        ColSep = ' | ';
        
        % Flag that determines if the first row should be used as table header.
        %
        % If true, a separate line with dashes will be inserted after the first printed row.
        %
        % @default false @type logical
        HasHeader = false;
        
        % Flag that determines if there is a row header for the table.
        %
        % If true, this causes a special behaviour regarding
        % formatting the the first argument passed to PrintTable.addRow.
        % Let `n` be the total number of row content arguments.
        % - No format is given: The first argument will be "stringyfied" as
        % all other elements
        % - A cell with `n=1` format strings, i.e. {'%f'}, is given: This
        % format will be applied to all but the first argument of addRow.
        % This is the actual reason why this flag was introduced, in order
        % to be able to specify a current row string header and set the
        % format for all other entries using one format string.
        % - A cell with `n-1` format strings is given: They will be applied
        % to the 2nd until last argument of addRow (in that order)
        % - A cell with `n` format strings is given: They will be applied
        % to all content arguments including the row header (in that order)
        %
        % @type logical @default false
        HasRowHeader = false;
        
        % The output format of the table when using the print method.
        %
        % Currently the values 'txt' for plaintext and 'tex' for LaTeX
        % output are available.
        %
        % @default 'txt' @type enum<'txt', 'tex'>
        Format = 'txt';
        
        % A caption for the table.
        %
        % Depending on the output the caption is added: 
        % plain: First line above table
        % tex: Inserted into the \\caption command
        %
        % @default '' @type char
        Caption = '';
        
        % Flag that indicates if exported tables in PDF format should be
        % sized to the actual table size or be contained in a normal
        % unformatted article page.
        %
        % @type logical @default true
        TightPDF = true;
        
        % Flag that determines if inserted tab characters are stripped from any string to
        % preserve the correct display of the PrintTable.
        %
        % Affects both directly set strings and those produced by providing a format pattern.
        %
        % @type logical @default true
        StripInsertedTabChars = true;
    end
    
    properties(Dependent)
        % The number of rows in the current table
        %
        % @type integer
        NumRows;
        
        % Flag that tells the PrintTable to surround numerical values with dollar signs "$$"
        % to obtain a more suitable LaTeX number representation.
        %
        % Set to false to produce plain output of cell contents "as they are".
        %
        % @type logical @default true
        TexMathModeDetection;
    end
    
    properties(SetAccess=private)
        % The string cell data
        data;
        
        % Flag matrix to indicate if a cell value is numeric or not.
        %
        % Used for TeX output.
        %
        % See also: TexMathModeDetection
        mathmode;
        
        % Maximum content length for each colummn
        contlen;
        
        % Flag to maximize user convenience; this is set
        % whenever a row adding resulted in a string containing a backslash, indicating that
        % latex commands are used. If so, a warning is issued on tex export if
        % TexMathModeDetection is switched off.
        haslatexcommand = false;
    end
    
    properties(Access=private)
        fTexMMode = true;
    end
    
    methods
        function this = PrintTable(caption, varargin)
            % Creates a new PrintTable instance.
            %
            % Parameters:
            % caption: The caption of the table. @type char @default ''
            % varargin: If given, they will be passed to sprintf using the
            % specified caption argument to create the table's Caption.
            % @type cell @default []
            this.clear;
            if nargin > 0
                if ~isempty(varargin)
                    this.Caption = sprintf(caption,varargin{:});
                else
                    this.Caption = caption;
                end
            end
        end
        
        function display(this)
            % Overload for the default builtin display method.
            %
            % Calls print with argument 1, i.e. standard output.
            for i = 1:length(this)
                this(i).print(1);
            end
        end
        
        function print(this, outfile)
            % Prints the current table to a file pointer.
            %
            % Parameters:
            % outfile: The file pointer to print to. Must either be a valid MatLab 'fileID'
            % or can be 1 or 2, for stdout or stderr, respectively. @type integer @default 1
            %
            % See also: fprintf
            if nargin == 1
                outfile = 1;
            end
            for i = 1:length(this)
                t = this(i);
                if strcmp(t.Format,'txt')
                    t.printPlain(outfile);
                elseif any(strcmp(t.Format,{'tex','pdf'}))
                    t.printTex(outfile);
                else
                    error('Unsupported format: %s',t.Format);
                end
            end
        end
        
        function str = toString(this)
            % Returns a character array representation of the current table.
            %
            % Return values:
            % str: The table as string @type char
            f = fopen('tmpout','w+');
            this.print(f);
            fclose(f);
            str = fileread('tmpout');
            delete('tmpout');
        end
        
        function saveToFile(this, filename, openfile)
            % Prints the current table to a file.
            %
            % If a file name is specified, the format is determined by the
            % file extension. Allowed types are "txt" for plain text, "tex"
            % for a LaTeX table and "pdf" for an immediate export of the
            % LaTeX table to a PDF document.
            %
            % @note The last option required "pdflatex" to be available on
            % the system's environment.
            %
            % Parameters:
            % filename: The file to print to. If the file exists, any
            % contents are discarded.
            % If the file does not exist, an attempt to create a new one is
            % made. @type char @default Prompts for saving target
            % openfile: Flag that indicates if the exported file should be
            % opened after saving. @type logical @default false
            
            % Case 1: No file name given
            if nargin < 2 || isempty(filename)
                initdir = getpref('PrintTable','LastDir',pwd);
                choices = {'*.txt', 'Text files (*.txt)';...
                           '*.tex', 'LaTeX files (*.tex)';...
                           '*.pdf', 'PDF files (*.pdf)'};
                [fname, path, extidx] = uiputfile(choices, ...
                    sprintf('Save table "%s" as',this.Caption), initdir);
                % Abort if no file was selected
                if fname == 0
                    return;
                end
                setpref('PrintTable','LastDir',path);
                ext = choices{extidx,1}(2:end);
                filename = fullfile(path, fname);
                % Take off the extension of the fname (automatically added
                % by uiputfile)
                fname = fname(1:end-4);
            % Case 2: File name given. Determine format by file
            % extension
            else
                [path, fname, ext] = fileparts(filename);
                if isempty(ext)
                    ext = ['.' this.Format];
                elseif ~any(strcmp(ext,{'.txt','.tex','.pdf'}))
                    error('Valid file formats are *.txt, *.tex, *.pdf');
                end
                if isempty(path)
                    path = '.';
                end
                % Try to create directory if not existing
                if ~isempty(path) && exist(path,'dir') ~= 7
                    mkdir(path);
                end
            end
            if nargin < 3
                openfile = false;
            end
            
            oldfmt = this.Format; % store old format
            this.Format = ext(2:end);
            % PDF export
            if strcmp(ext,'.pdf')
                this.pdfExport(path, fname);
            % Text export in either plain or LaTeX format
            else
                fid = fopen(filename,'w');
                this.print(fid);
                fclose(fid);
            end
            this.Format = oldfmt; % restore old format
            if openfile
                open(filename);
            end
        end
        
        function addMatrix(this, titles, data, varargin)
            for k=1:size(data,1)
                hlp = num2cell(data(k,:));
                this.addRow(titles{k},hlp{:},varargin{:});
            end    
        end
        
        function addRow(this, varargin)
            % Adds a row to the current table.
            %
            % Parameters:
            % varargin: Any number of arguments >= 1, each corresponding to a column of the
            % table. Each argument must be a char array.
            if isempty(varargin)
                error('Not enough input arguments.');
            end
            hasformat = iscell(varargin{end});
            if iscell(varargin{1})
                error('Invalid input argument. Cells cannot be added to the PrintTable, and if you wanted to specify a sprintf format you forgot the actual value to add.');
%             elseif hasformat && length(varargin)-1 ~= length(varargin{end})
%                 error('Input argument mismatch. If you specify a format string cell the number of arguments (=%d) to add must equal the number of format strings (=%d).',length(varargin)-1,length(varargin{end}));
            end
            if isempty(this.data)
                [this.data{1}, this.mathmode] = this.stringify(varargin);
                this.contlen = ones(1,length(this.data{1}));
            else
                % Check new number of columns
                newlen = length(varargin);
                if hasformat
                    newlen = newlen-1;
                end
                if length(this.data{1}) ~= newlen 
                    error('Inconsistent row length. Current length: %d, passed: %d',length(this.data{1}),newlen);
                end
                % Add all values
                [this.data{end+1}, this.mathmode(end+1,:)] = this.stringify(varargin);
            end
            % Record content length while building the table
            this.updateContentLengthAt(length(this.data));
            % Check if a latex command might be contained
            % (so far no re-computation is done on row removal)
            this.haslatexcommand = this.haslatexcommand ...
                || any(cellfun(@(s)~isempty(strfind(s,'\')),this.data{end}));
        end
        
%         function insertRow(this, pos, varargin)
%             this.addRow(varargin{:});
%             tmp = this.data{pos};
%             this.data{pos} = this.data{end};
%             this.data{end} = tmp;
%             tmp = this.mathmode(pos,:);
%             this.mathmode(pos,:) = this.mathmode(end,:);
%             this.mathmode(end,:) = tmp;
%         end
        
        function clear(this)
            % Clears the current PrintTable contents and caption.
            this.data = {};
            this.contlen = [];
            this.Caption = '';
        end
        
        function removeRow(this, idx)
            % Removes a row from the PrintTable
            %
            % Parameters:
            % idx: The index of the row to remove @type integer
            if idx < 1 || idx > length(this.data)
                error('Invalid row index: %d',idx);
            end
            this.data(idx) = [];
            this.updateContentLengths;
        end
        
        function this = sort(this, colnr, direction)
            % Sorts this table in alphanumeric order.
            %
            % Optionally, a column number and sort direction can be
            % specified.
            %
            % The default sort column is one. If you set HasHeader to true,
            % the default sort column will be two. However, the column
            % numbering will always include the first column in order to
            % explicitly allow sorting by row headers.
            %
            % Parameters:
            % colnr: The column number @type integer @default 1 or 2
            % depending on HasHeader setting
            % direction: The sort direction 'ascend' or 'descend' @type
            % char @default 'ascend'
            if nargin < 3
                direction = 'ascend';
                if nargin < 2
                    if this.HasHeader
                        colnr = 2;
                    else
                        colnr = 1;
                    end
                end
            end
            
            if isempty(this.data)
                return;
            elseif colnr < 1 || colnr > length(this.data{1})
                error('Please specify a valid column number');
            end
            
            vals = {};
            for k = 1:length(this.data)
                vals{k} = this.data{k}{colnr};%#ok
            end
            [~, sidx] = sort(vals);
            % "Manually" choose direction
            if strcmpi(direction,'descend')
                sidx = fliplr(sidx);
            end
            copy = this.data;
            copym = this.mathmode;
            for k = 1:length(this.data)
                copy{k} = this.data{sidx(k)};
                copym(k,:) = this.mathmode(sidx(k),:);
            end
            this.data = copy;
            this.mathmode = copym;
        end
        
        function transposed = ctranspose(this)
            transposed = this.clone;
            hlp = reshape([this.data{:}],length(this.data{1}),[]);
            transposed.data = {};
            for k=1:size(hlp,1)
                transposed.data{k} = hlp(k,:);
            end
            transposed.contlen = cellfun(@(row)max(cellfun(@(el)length(el),row)),...
                this.data);
            % Trigger correct content length guessing if need be
            transposed.TexMathModeDetection = this.fTexMMode;
            transposed.mathmode = this.mathmode';
        end
        
        function copy = clone(this)
            % Returns a new instance of PrintTable with the same content
            copy = PrintTable(this.Caption);
            copy.ColSep = this.ColSep;
            copy.HasHeader = this.HasHeader;
            copy.HasRowHeader = this.HasRowHeader;
            copy.Format = this.Format;
            copy.TightPDF = this.TightPDF;
            copy.data = this.data;
            copy.contlen = this.contlen;
            copy.mathmode = this.mathmode;
            copy.fTexMMode = this.fTexMMode;
            copy.haslatexcommand = this.haslatexcommand;
            copy.StripInsertedTabChars = this.StripInsertedTabChars;
        end
        
        function joined = append(this, table, columns)
            % Appends the cell contents from another table to this table.
            %
            % Parameters:
            % table: The other table @type PrintTable
            % columns: The column indices to transfer. @type rowvec<integer> @default all
            if isempty(this.data)
                error('No table rows exists to append to. Why would you want to do that?');
            end
            
            joined = this.clone;
            % Check data compatibility
            if ~isa(table,'PrintTable')
                error('Argument must be a PrintTable instance');
            end
            if table.NumRows == 0
                return;
            end
            if nargin < 3
                columns = 1:length(table.data{1});
            end
            if length(this.data{1}) ~= length(columns)
                error('Invalid column number: Have %d but want to append %d',length(this.data{1}),length(columns));
            end
            % Augment data
            start = 1;
            if table.HasHeader
                start = 2;
            end
            for k=start:length(table.data)
                joined.data{end+1} = table.data{k}(columns);
            end
            joined.contlen = max(this.contlen,table.contlen(columns));
            joined.mathmode = [this.mathmode; table.mathmode(:,columns)];
            joined.haslatexcommand = this.haslatexcommand || table.haslatexcommand;
            % Transfer Caption if not set locally
            if ~isempty(joined.Caption)
                joined.Caption = table.Caption;
            end
        end
        
        function set.ColSep(this, value)
            if ~isempty(value) && ~isa(value,'char')
                error('ColSep must be a char array.');
            end
            this.ColSep = value;
        end
    end
    
    %% Getter & Setter
    methods
        function set.HasHeader(this, value)
            if ~islogical(value) || ~isscalar(value)
                error('HasHeader must be a logical scalar.');
            end
            this.HasHeader = value;
        end
        
        function set.HasRowHeader(this, value)
            if ~islogical(value) || ~isscalar(value)
                error('HasRowHeader must be a logical scalar.');
            end
            this.HasRowHeader = value;
        end
        
        function set.Caption(this, value)
            if ~isempty(value) && ~ischar(value)
                error('Caption must be a character array.');
            end
            this.Caption = value;
        end
        
        function set.Format(this, value)
            % 'Hide' the valid format pdf as its only used internally .. i
            % know .. lazyness :-)
            if ~any(strcmp({'txt','tex','pdf'},value))
                error('Format must be either ''txt'' or ''tex''.');
            end
            this.Format = value;
	   end
	   
	   function set.TabCharLen(this, value)
		  if isscalar(value) && value > 0 && round(value) == value
			 this.TabCharLen = value;
		  else
			 error('Invalid argument for TabCharLen. Must be a positive integer scalar.');
		  end
	   end
	   
        function value = get.NumRows(this)
            value = length(this.data);
            if this.HasHeader
                value = max(0,value-1);
            end
        end
        
        function set.TexMathModeDetection(this, value)
            if this.fTexMMode ~= value
                if ~islogical(value) || ~isscalar(value)
                    error('TexMathModeDetection must be true or false.');
                end
                this.fTexMMode = value;
                this.updateContentLengths;
            end
        end
        
        function value = get.TexMathModeDetection(this)
            value = this.fTexMMode;
        end
    end
    
    %% Internal helpers
    methods(Access=private)
        
        function updateContentLengthAt(this, idx)
            fun = @(str,num)length(str) + 2*num; %2 characters for local $ tex environment
            ismm = num2cell(this.mathmode(idx,:));
            this.contlen = max([this.contlen; cellfun(fun,this.data{idx},ismm)]);
        end
        
        function updateContentLengths(this)
            if ~isempty(this.data)
                this.contlen = ones(1,length(this.data{1}));
                for idx = 1:length(this.data)
                    this.updateContentLengthAt(idx);
                end
            end
        end
        
        function printPlain(this, outfile)
            % Prints the table as plain text
            if ~isempty(this.Caption)
                fprintf(outfile,'Table ''%s'':\n',this.Caption);
            end
            for ridx = 1:length(this.data)
                this.printRow(ridx,outfile,this.ColSep);
                fprintf(outfile,'\n');
                if ridx == 1 && this.HasHeader
                    % Compute number of tabs
                    ttabs = 0;
                    for i = 1:length(this.data{ridx})
                        ttabs = ttabs +  ceil((length(this.ColSep)*(i~=1)+this.contlen(i))/this.TabCharLen);
                    end
                    fprintf(outfile,'%s\n',repmat('_',1,(ttabs+1)*this.TabCharLen));
                end
            end
        end
        
        function printTex(this, outfile)
            % Prints the table in LaTeX format
            
            if ~this.TexMathModeDetection && this.haslatexcommand
                warning('PrintTable:TexExport',...
                    'No TexMathModeDetection enabled but LaTeX commands have been detected. Export might produce invalid LaTeX code.');
            end

            % Add verbose comment
            if ~isempty(this.Caption)
                fprintf(outfile,'%% PrintTable "%s" generated on %s\n',this.Caption,datestr(clock));
            else
                fprintf(outfile,'%% PrintTable generated on %s\n',datestr(clock));
            end
            d = dbstack;
            d = d(find(~strcmp({d(:).file},'PrintTable.m'),1));
            if ~isempty(d)                
                fprintf(outfile,'%% Created in %s:%d at %s\n',d.name,d.line,which(d.file));
            end
            % Add an informative comment to make the user aware of it's options :-)
            fprintf(outfile,'%% Export settings: TexMathModeDetection %d, HasHeader %d, HasRowHeader %d, StripInsertedTabChars %d, IsPDF %d, TightPDF %d\n',...
                    this.TexMathModeDetection,this.HasHeader,this.HasRowHeader,this.StripInsertedTabChars,...
                    strcmp(this.Format,'pdf'),this.TightPDF);
            cols = 0;
            if ~isempty(this.data)
                cols = length(this.data{1});
            end
            % Only add surroundings for pure tex output or full-sized PDF
            % generation
            if strcmp(this.Format,'tex') || ~this.TightPDF
                fprintf(outfile,'\\begin{table}[!hb]\n\t\\centering\n\t\\def\\arraystretch{1.3}\n\t');
            elseif ~isempty(this.Caption)
                % Enable this if you want, but i found no straight way of putting the caption
                % above the table (for the given time&resources :-))
                %fprintf(outfile,'Table: %s\n',this.Caption);
            end
            if this.HasRowHeader
                aligns = ['r' repmat('l',1,cols-1)];
            else
                aligns = repmat('l',1,cols);
            end
            fprintf(outfile,'\\begin{tabular}{%s}\n',aligns);
            % Print all rows
            for ridx = 1:length(this.data)
                fprintf(outfile,'\t\t');
                this.printRow(ridx,outfile,'& ');
                fprintf(outfile,'\\\\\n');
                if ridx == 1 && this.HasHeader
                    fprintf(outfile,'\t\t\\hline\\\\\n');
                end
            end
            fprintf(outfile, '\t\\end{tabular}\n');
            % Only add surroundings for pure tex output or full-sized PDF
            % generation
            if strcmp(this.Format,'tex') || ~this.TightPDF
                if ~isempty(this.Caption)
                    fprintf(outfile,'\t\\caption{%s}\n',this.Caption);
                end
                fprintf(outfile, '\\end{table}\n');
            end
        end
        
        function pdfExport(this, path, fname)
            [status, msg] = system('pdflatex --version');
            if status ~= 0
                error('pdfLaTeX not found or not working:\n%s',msg);
            else
                cap = 'table';
                if ~isempty(this.Caption)
                    cap = ['''' this.Caption ''''];
                end
                fprintf('Exporting %s to PDF using "%s"...\n',cap,msg(1:strfind(msg,sprintf('\n'))-1));
            end

            texfile = fullfile(path, [fname '.tex']);
            fid = fopen(texfile,'w');
            fprintf(fid,'\\documentclass{article}\n\\begin{document}\n');
            if this.TightPDF
                fprintf(fid,'\\newsavebox{\\tablebox}\n\\begin{lrbox}{\\tablebox}\n');
            else
                fprintf(fid, '\\thispagestyle{empty}\n');
            end
            % Print actual tex table
            this.print(fid);
            if this.TightPDF
                fprintf(fid, ['\\end{lrbox}\n\\pdfhorigin=0pt\\pdfvorigin=0pt\n'...
                    '\\pdfpagewidth=\\wd\\tablebox\\pdfpageheight=\\ht\\tablebox\n'...
                    '\\advance\\pdfpageheight by \\dp\\tablebox\n'...
                    '\\shipout\\box\\tablebox\n']);
            end
            fprintf(fid,'\\end{document}');
            fclose(fid);
            [status, msg] = system(sprintf('pdflatex -interaction=nonstopmode -output-directory="%s" %s',path,texfile));%#ok
            if 0 ~= status
                delete(fullfile(path, [fname '.pdf']));
                fprintf(2,'PDF export failed, pdflatex finished with errors. See the <a href="matlab:edit(''%s'')">LaTeX logfile</a> for details.\n',fullfile(path, [fname '.log']));
            else
                delete(fullfile(path, [fname '.log']));
                fprintf('done!\n');
            end
            delete(texfile,fullfile(path, [fname '.aux']));
        end
        
        function printRow(this, rowidx, outfile, sep)
            % Prints a table row using a given separator whilst inserting appropriate amounts
            % of tabs
            row = this.data{rowidx};
            % Check if mathmode has been determined
            ismm = this.mathmode(rowidx,:);
            % Check if we are producing tex-based output
            istex = any(strcmp(this.Format,{'tex', 'pdf'}));
            sl = length(sep);
            for i = 1:length(row)-1
                str = row{i};
                if istex && this.fTexMMode && ismm(i)
                    str = ['$' str '$'];%#ok
                end
                fillstabs = floor((sl*(i~=1)+length(str))/this.TabCharLen);
                tottabs = ceil((sl*(i~=1)+this.contlen(i))/this.TabCharLen);
                fprintf(outfile,'%s%s',[str repmat(char(9),1,tottabs-fillstabs)],sep);
            end
            str = row{end};
            if istex && this.fTexMMode && ismm(end)
                str = ['$' str '$'];
            end
            fprintf(outfile,'%s',str);
        end
        
        function [str, ismm] = stringify(this, data)
            % Converts any datatype to a string in a suitable way for table display
            
            % Format cell array given
            if iscell(data{end})
                % if format cell is only one item but have more values,
                % apply same format string
                if length(data{end}) == 1 && length(data)-1 > 1
                    data{end} = repmat(data{end},1,length(data)-1);
                    if this.HasRowHeader
                        % Make sure the row header becomes a string if not
                        % already one
                        data(1) = this.stringify(data(1));
                        data{end}{1} = '%s';
                    end
                elseif this.HasRowHeader && length(data{end}) == length(data)-2
                    % Make sure the row header becomes a string if not
                    % already one
                    data(1) = this.stringify(data(1));
                    data{end} = ['%s' data{end}(:)'];
                end
                str = cell(1,length(data)-1);
                % Apply sprintf pattern to each element
                for i=1:length(data)-1
                    if isa(data{end}{i},'function_handle')
                        if nargin(data{end}{i}) > 1
                            tmpstr = data{end}{i}(data{i},i);
                        else
                            tmpstr = data{end}{i}(data{i});
                        end
                    else
                        tmpstr = sprintf(data{end}{i},data{i});
                    end
                    tmpstr = strrep(tmpstr,char(10),'');
                    % Strip tab chars if set
                    if this.StripInsertedTabChars
                        str{i} = strrep(tmpstr,char(9),'');
                    else
                        str{i} = tmpstr;
                    end
                end
            else % convert to strings if no specific format is given
                str = cell(1,length(data));
                for i=1:length(data)
                    el = data{i};
                    if isa(el,'char')
                        el = strrep(el,char(10),'');
                        % Use char array directly and strip tab characters if desired
                        if this.StripInsertedTabChars
                            str{i} = strrep(el,char(9),'');
                        else
                            str{i} = el;
                        end
                    elseif isinteger(el)
                        if numel(el) > 1
                            str{i} = ['[' this.implode(el(:),', ','%d') ']'];
                        else
                            str{i} = sprintf('%d',el);
                        end
                    elseif isnumeric(el)
                        if numel(el) > 1
                            if isvector(el) && length(el) < 100
                                str{i} = ['[' this.implode(el(:),', ','%g') ']'];
                            else
                                str{i} = ['[' this.implode(size(el),'x','%d') ' ' class(el) ']'];
                            end
                        else
                            if isempty(el)
                                str{i} = '[]';
                            else
                                str{i} = sprintf('%g',el);
                            end
                        end
                    elseif isa(el,'function_handle')
                        str{i} = func2str(el);
                    elseif isa(el,'handle')
                        mc = metaclass(el);
                        str{i} = mc.Name;
                    elseif islogical(el)
                        if numel(el) > 1
                            str{i} = this.implode(el(:),', ','%d');
                        else
                            str{i} = sprintf('%d',el);
                        end
                    else
                        error('Cannot automatically convert an argument of type %s for PrintTable display.',class(el));
                    end
                end
            end
            % Detect if any of the cell contents are numerical values
            fun = @(arg)~isempty(arg) && arg(1) ~= '$' && arg(end) ~= '$' ...
                        && (~isnan(str2double(arg)) || ~isempty(strfind(arg,'\')));
            ismm = cellfun(fun,str);
        end
        
        function str = implode(this, data, glue, format)%#ok
            str = '';
            if ~isempty(data)
                if nargin < 3
                    format = '%2.3e';
                    if nargin < 2
                        glue = ', ';
                    end
                end
                if isa(data,'cell')
                    str = data{1};
                    for idx = 2:length(data)
                        str = [str glue data{idx}];%#ok
                    end
                elseif isnumeric(data)
                    % first n-1 entries
                    if numel(data) > 1
                        str = sprintf([format glue],data(1:end-1));
                    end
                    % append last, no glue afterwards needed
                    str = [str sprintf(format,data(end))];
                else
                    error('Can only pass cell arrays of strings or a vector with sprintf format pattern');
                end
            end
        end
    end
    
    %% Tests
    methods(Static)
        function [res, t] = test_PrintTable
            % A simple test for PrintTable
            t = PrintTable;
            t.Caption = 'This is my PrintTable test.';
            t.addRow('A','B','C');
            t.addRow('123',456E10,'789');
            t.addRow(1234567,'12345','789');
            t.addRow('1234567',123456,'789');
            t.addRow('34','\sin(x)+4','somestring');
            t.addRow('foo','bar',datestr(clock));
            t.addRow(123.45678,pi,789,{'%2.3f','$%4.4g$','decimal: %d'});
            
            % If HasHeader is set, the format args can be omitted for the first column.
            t.HasRowHeader = true;
            t.addRow(123.45678,pi,789,{'$%4.4g$','decimal: %d'});
            t.HasRowHeader = false;
            
            t.addRow('12345678','\latexcommand{foo}+5','789');
            t.addRow('yet ','123','789');
            t.addRow(123.45678,pi,789,{'%2.3f',@(v)sprintf('functioned pi=%g!',v-3),'decimal: %d'});
            t.addRow(2,2,2,{@(v,colidx)sprintf('callback using column index: %d!',v^colidx)});
            t.addRow('single',pi,'fun with colidx',{'%s',@(v,colidx)sprintf('value at column %d: %g!',colidx,v),'%s'});
            t.addRow('next col with tabs',char(9),'StripInsertedTabChars on',{'%s','>\t>%s>\t>','%s'});
            
            fprintf(2,'test_PrintTable: Plain text display:\n');
            t.display;
            
            fprintf(2,'test_PrintTable: Plain text display with no tab stripping:\n');
            t.StripInsertedTabChars = false;
            t.addRow('next col with tabs',char(9),'StripInsertedTabChars off (destroys layout)',{'%s','>\t>%s>\t>','%s'});
            t.display;
            t.removeRow(t.NumRows);
            
            fprintf(2,'test_PrintTable: Plain text display with header:\n');
            t.HasHeader = true;
            t.display;
            
            fprintf(2,'test_PrintTable: LaTeX display:\n');
            t.HasHeader = false;
            t.Format = 'tex';
            t.print;
            
            fprintf(2,'test_PrintTable: LaTeX display with header:\n');
            t.HasHeader = true;
            t.print;
            t.HasHeader = false;
            
            fprintf(2,'test_PrintTable: LaTeX display and no TexMathModeDetection:\n');
            t.TexMathModeDetection = false;
            t.print;
            t.TexMathModeDetection = true;
            
            fprintf(2,'test_PrintTable: LaTeX display of transposed:\n');
            tt = t';
            tt.print;
            
            fprintf(2,'test_PrintTable: Plain text display of transposed:\n');
            tt.Format = 'txt';
            tt.print;
            res = true;
        end
        
        function [res, t] = test_PrintTable_Misc
            % A simple test for PrintTable
            t = PrintTable('This is PrintTable Misc Features test, created on %s',datestr(now));
            t.HasRowHeader = true;
            t.HasHeader = true;
            t.addRow('A','B','C');
            t.addRow('header-autofmt',456,789,{'%d'});
            t.addRow(1234.345,456,789,{'%2.2E','%d','%d'});
            t.addRow('header-expl-fmt',456,pi,{'%s','%d','%2.2f'});
            t.addRow('nofmt-header',456,pi,{'%d','%f'});
            t.addRow('1234567','12345','789');
            t.addRow('foo','bar',datestr(clock));
            t.addRow(123.45678,pi,789,{'%2.3f','%4.4g','decimal: %d'});
            t.addRow(12345678,'123','789');
            t.addRow(12345.6789,'123','789');
            t.display;
            
            fprintf(2,'test_PrintTable_Misc: Tex-Format:\n');
            t.Format = 'tex';
            t.print;
            
            fprintf(2,'test_PrintTable_Misc: Tex-Format with Row Header:\n');
            t.HasHeader = true;
            t.print;
            
            fprintf(2,'test_PrintTable_Misc: Self-Appended:\n');
            t.Format = 'txt';
            ta = t.append(t);
            ta.display;
            
            t2 = PrintTable('dummy');
            t2.addRow('A','B','C','D','E');
            t2.addRow('header1',456,789,1,34,{'%d'});
            t2.addRow(12345.6789,'123','789','foo','bar');
            t2.addRow('header2',4656,35789,351,4);
            
            fprintf(2,'test_PrintTable_Misc: Test table to append:\n');
            t2.display;
            
            fprintf(2,'test_PrintTable_Misc: Appending some test table columns (with header):\n');
            ta = t.append(t2,[1 3 5]);
            ta.display;
            
            fprintf(2,'test_PrintTable_Misc: Appending some test table columns (without header):\n');
            t2.HasHeader = true;
            ta = t.append(t2,[4 2 1]);
            ta.display;
            
            % Empty table append test
            t2.clear;
            t.append(t2);
            
            res = true;
        end
        
        function [res, t] = test_PrintTable_LaTeX_Export
            % A simple test for PrintTable
            t = PrintTable('LaTeX/PDF export demo, %s',datestr(now));
            t.HasRowHeader = true;
            t.HasHeader = true;
            t.addRow('A','B','C');
            t.addRow('Data 1',456,789,{'%d'});
            t.addRow(1234.345,456,789,{'$%2.2E$','$%d$','$%d$'});
            t.addRow('header-expl-fmt',456,pi,{'%s','%d','%2.2f'});
            t.addRow('1234567','12345','789');
            x = 4;
            t.addRow('x=4','\sin(x)',sin(x),{'%s','%f'});
            % Explicit wrapping directly or with $$ per format string
            t.addRow('x=4','25 (5*5)=',5*5,{'$%s$','%d'});
            % (have HasRowHeader, thus the first format string is not necessary (but yet
            % possible to define))
            t.addRow('$x=4,\alpha=.2$','\alpha\exp(x)\cos(x)',.2*exp(x)*cos(x),{'$%s$','%f'});
            t.display;
            
            % LaTeX
            t.saveToFile('myLaTeXexportedPrintTable.tex',true);
            
            % Tight PDF
            t.saveToFile('mytable_tightpdf.pdf',true);
            
            % Whole page PDF
            t.TightPDF = false;
            t.TexMathModeDetection = true;
            t.saveToFile('mytable_fullpage.pdf',true);
            res = true;
        end
        
        function [res, t] = test_PrintTable_Failed_LaTeX_Export
            t = PrintTable('LaTeX/PDF export demo (THIS MUST FAIL!), %s',datestr(now));
            t.HasRowHeader = true;
            t.HasHeader = true;
            t.TexMathModeDetection = false;
            t.addRow('A','B','C');
            t.addRow('Data 1',456,789,{'%d'});
            t.addRow(1234.345,456,789,{'$%2.2E$','$%d$','$%d$'});
            t.addRow('header-expl-fmt',456,pi,{'%s','%d','%2.2f'});
            t.addRow(1234567,'12345',45.3463E-4);
            x = 4;
            % This is the offending line
            t.addRow('x=4','\sin(x)',sin(x),{'%s','%f'});
            % Explicit wrapping directly or with $$ per format string
            t.addRow('x=4','25 (5*5)=',5*5,{'$%s$','%d'});
            t.addRow('$x=4,\alpha=.2$','\alpha\exp(x)\cos(x)',.2*exp(x)*cos(x),{'$%s$','%f'});
            t.display;
            try
                t.saveToFile('mytable_tightpdf_nodetection.pdf',true);
            catch ME
                fprintf(2,'TEST FAILED AS EXPECTED, WARNING HAS BEEN ISSUED. ALL GOOD.\n');
                res = true;
                return;
            end
            res = false;
        end
    end
    
    methods(Static,Access=private)
        function l = getDefaultTabCharLen
            % Computes the default value for equivalent length of a tab character in
            % single-space characters.
            %
            % Defining it here avoids having to adjust the TabCharLen at every instance when the
            % shipped default is not matching.
            %
            % Change this method to return whatever TabCharLength is default to your system if
            % the automatic detection does not work, however, please share your
            % situation/solution with me so i can include it here. Thanks!
            
            % Use default value 8
            defaultLen = 8;
            
            jDesktop = com.mathworks.mde.desk.MLDesktop.getInstance;
            % Display of output in command window
            if ~isempty(jDesktop.getClient('Command Window'))
                % Display of output in console etc
                l = com.mathworks.services.Prefs.getIntegerPref('CommandWindowSpacesPerTab',defaultLen);
            elseif isunix
                [c, l] = system('echo -n $''\t'' | wc -L');
                if c == 0
                    l = str2double(l);
                else
                    l = defaultLen;
                end
            else
                % Not checked yet for windows command line, but i guess this will be a rare
                % occasion. Please share if you encounter this situation.
                l = defaultLen;
            end
        end
    end
end