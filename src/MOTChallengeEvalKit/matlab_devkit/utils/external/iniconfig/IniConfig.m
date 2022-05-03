classdef IniConfig < handle
    %IniConfig - The class for working with configurations of settings and INI-files. 
    % This class allows you to create configurations of settings, and to manage them. 
    % The structure of the storage settings is similar to the structure of 
    % the storage the settings in the INI-file format. 
    % The class allows you to import settings from the INI-file and to export 
    % the settings in INI-file. 
    % Can be used for reading/writing data in the INI-file and managing 
    % settings of application.
    %
    %
    % Using:
    %   ini = IniConfig()
    %
    % Public Properties:
    %   Enter this command to get the properties:
    %   >> properties IniConfig
    %
    % Public Methods:
    %   Enter this command to get the methods:
    %   >> methods IniConfig
    %
    %   Enter this command to get more info of method:
    %   >> help IniConfig/methodname
    %
    %
    % Config Syntax:
    %
    %   ; some comments
    %
    %   [Section1] ; allowed the comment to section
    %   ; comment on the section
    %   key1 = value_1 ; allowed a comment to an individual parameter
    %   key2 = value_2
    %   
    %   [Section2]
    %   key1 = value_1.1, value_1.2     ; array data
    %   key2 = value_2
    %   ...
    %
    % Note:
    %   * There may be spaces in the names of sections and keys
    %   * Keys should not be repeated in the section (will read the last)
    %   * Sections should not be repeated in the config (will read the last)
    %
    % Supported data types:
    %   * numeric scalars and vectors
    %   * strings
    %
    %
    % Example:
    %   ini = IniConfig();
    %   ini.ReadFile('example.ini')
    %   ini.ToString()
    %
    % Example:
    %   ini = IniConfig();
    %   ini.ReadFile('example.ini')
    %   sections = ini.GetSections()
    %   [keys, count_keys] = ini.GetKeys(sections{1})
    %   values = ini.GetValues(sections{1}, keys)
    %   new_values(:) = {rand()};
    %   ini.SetValues(sections{1}, keys, new_values, '%.3f')
    %   ini.WriteFile('example1.ini')
    %
    % Example:
    %   ini = IniConfig();
    %   ini.AddSections({'Some Section 1', 'Some Section 2'})
    %   ini.AddKeys('Some Section 1', {'some_key1', 'some_key2'}, {'hello!', [10, 20]})
    %   ini.AddKeys('Some Section 2', 'some_key3', true)
    %   ini.AddKeys('Some Section 2', 'some_key1')
    %   ini.WriteFile('example2.ini')
    %
    % Example:
    %   ini = IniConfig();
    %   ini.AddSections('Some Section 1')
    %   ini.AddKeys('Some Section 1', 'some_key1', 'hello!')
    %   ini.AddKeys('Some Section 1', {'some_key2', 'some_key3'}, {[10, 20], [false, true]})
    %   ini.WriteFile('example31.ini')
    %   ini.RemoveKeys('Some Section 1', {'some_key1', 'some_key3'})
    %   ini.RenameKeys('Some Section 1', 'some_key2', 'renamed_some_key2')
    %   ini.RenameSections('Some Section 1', 'Renamed Section 1')
    %   ini.WriteFile('example32.ini')
    %
    %
    % See also:
    %   textscan, containers.Map
    %
    %
    % Author:         Iroln <esp.home@gmail.com>
    % Version:        1.2
    % First release:  25.07.09
    % Last revision:  21.03.10
    % Copyright:      (c) 2009-2010 Evgeny Prilepin aka Iroln
    %
    % Bug reports, questions, etc. can be sent to the e-mail given above.
    %
    
    
    properties (GetAccess = 'public', SetAccess = 'private')
        comment_style = ';' % style of comments
        count_sections = 0  % number of sections
        count_all_keys = 0  % number of all keys
    end
    
    properties (GetAccess = 'private', SetAccess = 'private')
        config_data_array = {}
        indicies_of_sections
        indicies_of_empty_strings
        
        count_strings = 0
        count_empty_strings = 0
        
        is_created_configuration = false
    end
    
    
    %======================================================================
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %------------------------------------------------------------------
        function obj = IniConfig()
            %IniConfig - constructor
            % To Create new object with empty default configuration.
            %
            % Using:
            %   obj = IniConfig()
            %
            % Input:
            %   none
            %
            % Output:
            %   obj - an instance of class IniConfig
            % -------------------------------------------------------------
            
            obj.CreateIni();
        end
        
        %------------------------------------------------------------------
        function CreateIni(obj)
            %CreateIni - create new empty configuration
            %
            % Using:
            %   CreateIni()
            %
            % Input:
            %   none
            %
            % Output:
            %   none
            % -------------------------------------------------------------
            
            obj.config_data_array = cell(2,3);
            obj.config_data_array(:,:) = {''};
            
            obj.updateCountStrings();
            obj.updateSectionsInfo();
            obj.updateEmptyStringsInfo();
            
            obj.is_created_configuration = true;
        end
        
        %------------------------------------------------------------------
        function status = ReadFile(obj, file_name, comment_style)
            %ReadFile - to read in the object the config data from a INI file
            %
            % Using:
            %   status = ReadFile(file_name, comment_style)
            %
            % Input:
            %   file_name - INI file name
            %   comment_style - style of comments in INI file
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(2, 3, nargin));
            CheckIsString(file_name);
            
            if (nargin == 3)
                obj.comment_style = ValidateCommentStyle(comment_style);
            end
            
            % Get data from file
            [file_data, status] = GetDataFromFile(file_name);
            
            if (status)
                obj.count_strings = size(file_data, 1);
                
                obj.config_data_array = ...
                    ParseConfigData(file_data, obj.comment_style);
                
                obj.updateSectionsInfo();
                obj.updateEmptyStringsInfo();
                obj.updateCountKeysInfo();
                
                obj.is_created_configuration = true;
            end
        end
        
        %------------------------------------------------------------------
        function status = IsSections(obj, section_names)
            %IsSections - determine whether there is a sections
            %
            % Using:
            %   status = IsSections(section_names)
            %
            % Input:
            %   section_names - name of section(s)
            %
            % Output:
            %   status - 1 (true) - yes, 0 (false) - no
            % -------------------------------------------------------------
            
            error(nargchk(2, 2, nargin));
            
            section_names = DataToCell(section_names);
            
            section_names = cellfun(@(x) obj.validateSectionName(x), ...
                section_names, 'UniformOutput', false);
            
            status = cellfun(@(x) obj.isSection(x), ...
                section_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function [section_names, count_sect] = GetSections(obj)
            %GetSections - get names of all sections
            %
            % Using:
            %   section_names = GetSections()
            %   [names_sect, count_sect] = GetSections()
            %
            % Input:
            %   none
            %
            % Output:
            %   section_names - cell array with the names of sections
            %   count_sect - number of sections in configuration
            % -------------------------------------------------------------
            
            error(nargchk(1, 1, nargin));
            
            section_names = obj.config_data_array(obj.indicies_of_sections, 1);
%             section_names = strrep(section_names, '[', '');
%             section_names = strrep(section_names, ']', '');
            
            count_sect = obj.count_sections;
        end
        
        %------------------------------------------------------------------
        function status = AddSections(obj, section_names)
            %AddSections - add sections to end configuration
            %
            % Using:
            %   status = AddSections(section_names)
            %
            % Input:
            %   section_names - name of section
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(2, 2, nargin));
            
            section_names = DataToCell(section_names);
            
            section_names = cellfun(@(x) obj.validateSectionName(x), ...
                section_names, 'UniformOutput', false);
            
            status = cellfun(@(x) obj.addSection(x), ...
                section_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function status = InsertSections(obj, positions, section_names)
            %InsertSections - insert sections to given positions
            %
            % Using:
            %   status = InsertSections(positions, section_names)
            %
            % Input
            %   positions - positions of sections
            %   section_names - names of sections
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(3, 3, nargin));
            
            positions = DataToCell(positions);
            section_names = DataToCell(section_names);
            
            CheckEqualNumberElems(positions, section_names);
            
            section_names = cellfun(@(x) obj.validateSectionName(x), ...
                section_names, 'UniformOutput', false);
    
            status = cellfun(@(x, y) obj.insertSection(x, y), ...
                positions, section_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function status = RemoveSections(obj, section_names)
            %RemoveSections - remove given section
            %
            % Using:
            %   status = RemoveSections(section_names)
            %
            % Input:
            %   section_names - names of sections
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(2, 2, nargin));
            
            section_names = DataToCell(section_names);
            
            status = cellfun(@(x) obj.removeSection(x), ...
                section_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function status = RenameSections(obj, old_section_names, new_section_names)
            %RenameSections - rename given sections
            %
            % Using:
            %   status = RenameSections(old_section_names, new_section_names)
            %
            % Input:
            %   old_section_names - old names of sections
            %   new_section_names - new names of sections
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(3, 3, nargin));
            
            old_section_names = DataToCell(old_section_names);
            new_section_names = DataToCell(new_section_names);
            
            CheckEqualNumberElems(old_section_names, new_section_names);
            
            status = cellfun(@(x, y) obj.renameSection(x, y), ...
                old_section_names, new_section_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function status = IsKeys(obj, section_name, key_names)
            %IsKeys - determine whether there is a keys in a given section
            %
            % Using:
            %   status = IsKeys(section_name, key_names)
            %
            % Input:
            %   key_names - name of keys
            %
            % Output:
            %   status - 1 (true) - yes, 0 (false) - no
            % -------------------------------------------------------------
            
            error(nargchk(3, 3, nargin));
            
            key_names = DataToCell(key_names);
            
            section_name = obj.validateSectionName(section_name);
            section_names = PadDataToCell(section_name, numel(key_names));
            
            status = cellfun(@(x, y) obj.isKey(x, y), ...
                section_names, key_names, 'UniformOutput', 1);
        end
        
        %------------------------------------------------------------------
        function [key_names, count_keys] = GetKeys(obj, section_name)
            %GetKeys - get names of all keys from given section
            %
            % Using:
            %   key_names = GetKeys(section_name)
            %   [key_names, count_keys] = GetKeys(section_name)
            %
            % Input:
            %   section_name - name of section
            %
            % Output:
            %   key_names - cell array with the names of keys
            %   count_keys - number of keys in given section
            % -------------------------------------------------------------
            
            error(nargchk(2, 2, nargin));
            
            section_name = obj.validateSectionName(section_name);            
            [key_names, count_keys] = obj.getKeys(section_name);
        end
        
        %------------------------------------------------------------------
        function [status, tf_set_values] = ...
                AddKeys(obj, section_name, key_names, key_values, value_formats)
            %AddKeys - add keys in a end given section
            %
            % Using:
            %   status = AddKeys(section_name, key_names)
            %   status = AddKeys(section_name, key_names, key_values)
            %   status = AddKeys(section_name, key_names, key_values, value_formats)
            %   [status, tf_set_values] = AddKeys(...)
            %
            % Input:
            %   section_name -- name of section
            %   key_names -- names of keys
            %   key_values -- values of keys (optional)
            %   value_formats --
            %
            % Output:
            %   status -- 1 (true): Success, status - 0 (false): Failed
            %   tf_set_values -- 1 (true): Success, status - 0 (false): Failed
            % -------------------------------------------------------------
            
            error(nargchk(3, 5, nargin));
            
            key_names = DataToCell(key_names);
            num_of_keys = numel(key_names);
            
            if (nargin < 5)
                value_formats = PadDataToCell('', num_of_keys);
            end
            if (nargin < 4)
                key_values = PadDataToCell('', num_of_keys);
            end
            
            key_values = DataToCell(key_values);
            value_formats = DataToCell(value_formats);
            
            CheckEqualNumberElems(key_names, key_values);
            CheckEqualNumberElems(key_values, value_formats);
            
            key_values = ValidateValues(key_values);
            
            section_name = obj.validateSectionName(section_name);
            section_names = PadDataToCell(section_name, num_of_keys);
            
            [status, tf_set_values] = cellfun(@(a, b, c, d) obj.addKey(a, b, c, d), ...
                section_names, key_names, key_values, value_formats, 'UniformOutput', 1);
        end
        
        %------------------------------------------------------------------
        function [status, tf_set_values] = InsertKeys(obj, ...
                section_name, key_positions, key_names, key_values, value_formats)
            %InsertKeys - insert keys into the specified positions in a given section
            %
            % Using:
            %   status = InsertKeys(section_name, key_positions, key_names)
            %   status = InsertKeys(section_name, key_positions, key_names, key_values)
            %   status = InsertKeys(section_name, key_positions, key_names, key_values, value_formats)
            %   [status, tf_set_values] = InsertKeys(...)
            %
            % Input:
            %   section_name -- name of section
            %   key_positions -- positions of keys in section
            %   key_names -- names of keys
            %   key_values -- values of keys (optional)
            %   value_formats --
            %
            % Output:
            %   status - 1 (true): Success, status - 0 (false): Failed
            %   tf_set_values - 1 (true): Success, status - 0 (false): Failed
            % -------------------------------------------------------------
            
            error(nargchk(4, 6, nargin));
            
            key_positions = DataToCell(key_positions);
            key_names = DataToCell(key_names);
            num_of_keys = numel(key_names);
            
            CheckEqualNumberElems(key_positions, key_names);
            
            if (nargin < 6)
                value_formats = PadDataToCell('', num_of_keys);
            end
            if (nargin < 5)
                key_values = PadDataToCell('', num_of_keys);
            end
            
            key_values = DataToCell(key_values);
            value_formats = DataToCell(value_formats);
            
            CheckEqualNumberElems(key_names, key_values);
            CheckEqualNumberElems(key_values, value_formats);
            
            key_values = ValidateValues(key_values);
            
            section_name = obj.validateSectionName(section_name);
            section_names = PadDataToCell(section_name, num_of_keys);
            
            [status, tf_set_values] = ...
                cellfun(@(a, b, c, d, e) obj.insertKey(a, b, c, d, e), ...
                section_names, key_positions, key_names, ...
                key_values, value_formats, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function status = RemoveKeys(obj, section_name, key_names)
            %RemoveKeys - remove the keys from a given section
            %
            % Using:
            %   status = RemoveKeys(section_name, key_names)
            %
            % Input:
            %   section_name - name of section
            %   key_names - names of keys
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(3, 3, nargin));
            
            key_names = DataToCell(key_names);
            
            section_name = obj.validateSectionName(section_name);
            section_names = PadDataToCell(section_name, numel(key_names));
            
            status = cellfun(@(a, b) obj.removeKey(a, b), ...
                section_names, key_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function status = RenameKeys(obj, section_name, old_key_names, new_key_names)
            %RenameKeys - rename the keys in a given section
            %
            % Using:
            %   status = RenameKeys(section_name, old_key_names, new_key_names)  
            %
            % Input:
            %   section_name - name of section
            %   old_key_names - old names of keys
            %   new_key_names - new names of keys
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(4, 4, nargin));
            
            old_key_names = DataToCell(old_key_names);
            new_key_names = DataToCell(new_key_names);
            
            CheckEqualNumberElems(old_key_names, new_key_names);
            
            section_name = obj.validateSectionName(section_name);
            section_names = PadDataToCell(section_name, numel(old_key_names));
            
            status = cellfun(@(a, b, c) obj.renameKey(a, b, c), ...
                section_names, old_key_names, ...
                new_key_names, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function [values, status] = GetValues(obj, section_name, key_names, default_values)
            %GetValues - get values of keys from given section
            %
            % Using:
            %   values = GetValues(section_name, key_names)
            %   values = GetValues(section_name, key_names, default_values)
            %
            % Input:
            %   section_name -- name of given section
            %   key_names -- names of given keys
            %   default_values -- values of keys that are returned by default
            %
            % Output:
            %   values -- cell array with the values of keys
            %   status -- 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(2, 4, nargin));
            
%             if (nargin < 2)
%                 % get all values
%                 sections = obj.GetSections();
%                 
%                 values = {};
%                 status = [];
%                 for i = 1:numel(sections)
%                     [vals, tf] = obj.GetValues(sections{i});
%                     values = cat(1, values, vals);
%                     status = cat(1, status, tf);
%                 end
%                 return;
%             end
            
            section_name = obj.validateSectionName(section_name);
            
            if (nargin < 3)
                % get all values from given section
                key_names = obj.getKeys(section_name);
                if isempty(key_names)
                    values = {};
                    status = false;
                    return;
                end
            end
            
            if iscell(key_names)
                is_cell = true;
            else
                is_cell = false;
            end
            
            key_names = DataToCell(key_names);
            
            if (nargin < 4)
                default_values = PadDataToCell([], numel(key_names));
            end
            
            default_values = DataToCell(default_values);
            default_values = ValidateValues(default_values);
            
            CheckEqualNumberElems(key_names, default_values);
            
            section_names = PadDataToCell(section_name, numel(key_names));
            
            [values, status] = cellfun(@(x, y, z) obj.getValue(x, y, z), ...
                section_names, key_names, default_values, 'UniformOutput', false);
            
            if (~is_cell)
                values = values{1}; 
            end
            status = cell2mat(status);
        end
        
        %------------------------------------------------------------------
        function status = SetValues(obj, section_name, key_names, key_values, value_formats)
            %SetValues - set values for given keys from given section
            %
            % Using:
            %   status = SetValues(section_name, key_names, key_values)
            %   status = SetValues(section_name, key_names, key_values, value_formats)
            %
            % Input:
            %   section_name -- name of given section (must be string)
            %   key_names -- names of given keys (must be cell array of strings or string)
            %   key_values -- values of keys (must be cell array or one value)
            %   value_formats -- 
            %
            % Output:
            %   status -- 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(4, 5, nargin));     
            
            key_names = DataToCell(key_names);
            num_of_keys = numel(key_names);
            
            if (nargin < 5)
                value_formats = PadDataToCell('', num_of_keys);
            end
            
            key_values = DataToCell(key_values);
            value_formats = DataToCell(value_formats);

            CheckEqualNumberElems(key_names, key_values);
            CheckEqualNumberElems(key_values, value_formats);
            
            key_values = ValidateValues(key_values);
            
            section_name = obj.validateSectionName(section_name);
            section_names = PadDataToCell(section_name, num_of_keys);
            
            status = cellfun(@(a, b, c, d) obj.setValue(a, b, c, d), ...
                section_names, key_names, key_values, value_formats, 'UniformOutput', true);
        end
        
        %------------------------------------------------------------------
        function varargout = ToString(obj, section_name)
            %ToString - export configuration to string or display
            %
            % Using:
            %   ToString()
            %   ToString(section_name)
            %   str = ToString(...)
            %
            % Input:
            %   section_name - name of sections for export (optional)
            %
            % Output:
            %   str - string with full or section configuration (optional)
            % -------------------------------------------------------------
            
            %FIXME: возможно, нужно разбить этот метод на несколько, он слишком длинный 
            
            error(nargchk(1, 2, nargin));
            
            if (nargin < 2)
                is_full_export = true;
            else
                section_name = obj.validateSectionName(section_name);
                is_full_export = false;
            end
            
            if is_full_export
                count_str = obj.count_strings;
                indicies = 1:count_str;
            else
                first_index = getSectionIndex(obj, section_name);
                key_indicies = obj.getKeysIndexes(section_name);
                
                if isempty(key_indicies)
                    last_index = first_index;
                else
                    last_index = key_indicies(end);
                end
                
                indicies = first_index:last_index;
            end
            
            indicies_of_sect = obj.indicies_of_sections;
            config_data = obj.config_data_array;
            
            str = '';
            conf_str = sprintf('\n');
                
            for k = indicies
                if isempty(config_data{k,1})
                    if isempty(config_data{k,3})
                        str = sprintf('\n');
                    else
                        comment_str = config_data{k,3};
                        str = sprintf('%s\n', comment_str);
                    end
                    
                elseif ~isempty(indicies_of_sect(indicies_of_sect == k))
                    if is_full_export
                        if isempty(config_data{k,3})
                            section_str = config_data{k,1};
                            
                            str = sprintf('%s\n', section_str);
                        else
                            section_str = config_data{k,1};
                            comment_str = config_data{k,3};
                            
                            str = sprintf('%s    %s\n', ...
                                section_str, comment_str);
                        end
                    end
                    
                elseif ~isempty(config_data{k,1}) && ...
                        isempty(indicies_of_sect(indicies_of_sect == k))
                    if isempty(config_data{k,3})
                        key_str = config_data{k,1};
                        val_str = config_data{k,2};
                        
                        str = sprintf('%s=%s\n', key_str, val_str);
                        
                    else
                        key_str = config_data{k,1};
                        val_str = config_data{k,2};
                        comment_str = config_data{k,3};
                        
                        str = sprintf('%s=%s    %s\n', ...
                            key_str, val_str, comment_str);
                    end
                end
                
                conf_str = sprintf('%s%s', conf_str, str);
            end
            
            if (nargout == 0)
                fprintf(1, '%s\n', conf_str);
            elseif (nargout == 1)
                varargout{1} = conf_str(2:end);
            else
                error('Too many output arguments.')
            end
        end
        
        %------------------------------------------------------------------
        function status = WriteFile(obj, file_name)
            %WriteFile - write to the configuration INI file on disk
            %
            % Using:
            %   status = WriteFile(file_name)
            %
            % Input:
            %   file_name - name of output INI file
            %
            % Output:
            %   status - 1 (true) - success, 0 (false) - failed
            % -------------------------------------------------------------
            
            error(nargchk(2, 2, nargin));
            CheckIsString(file_name);
            
            fid = fopen(file_name, 'w');
            
            if (fid ~= -1)
                str = obj.ToString();
                fprintf(fid, '%s', str);
                
                fclose(fid);
                status = true;
            else
                status = false;
                return;
            end
        end
        
    end % public methods
    %----------------------------------------------------------------------
    
    
    %======================================================================
    methods (Access = 'private')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %------------------------------------------------------------------
        function num = nameToNumSection(obj, section_name)
            %nameToNumSection - get number of section
            
            section_names = obj.GetSections();
            int_ind = find(strcmp(section_names, section_name));
            
            if ~isempty(int_ind)
                % If the section is not unique, then choose the latest
                num = int_ind(end);
            else
                num = [];
            end
        end
        
        %------------------------------------------------------------------
        function sect_index = getSectionIndex(obj, section_name)
            %getSectionIndex - get index of section in config data
            
            num = obj.nameToNumSection(section_name);
            sect_index = obj.indicies_of_sections(num);
        end
        
        %------------------------------------------------------------------
        function [key_indicies, count_keys] = getKeysIndexes(obj, section_name)
            %getKeysIndexes - get keys indices from given section
            
            sect_num = obj.nameToNumSection(section_name);
                
            if isempty(sect_num)
                key_indicies = [];
                count_keys = 0;
                return;
                
            elseif (sect_num == obj.count_sections)
                key_indicies = ...
                    obj.indicies_of_sections(sect_num)+1:obj.count_strings;
            else
                key_indicies = ...
                    obj.indicies_of_sections(sect_num)+1:obj.indicies_of_sections(sect_num+1)-1;
            end
            
            indicies_of_empty = obj.indicies_of_empty_strings;
            empty_indicies = ismember(key_indicies, indicies_of_empty);
            
%             empty_indicies = cellfun('isempty', ...
%                 obj.config_data_array(key_indicies, 1));
            
            key_indicies(empty_indicies) = [];
            key_indicies = key_indicies(:);
            count_keys = length(key_indicies);
        end
        
        %------------------------------------------------------------------
        function ind = getKeyIndex(obj, section_name, key_name)
            %getKeyIndex - get key index
            
            key_names = obj.getKeys(section_name);
            key_indicies = obj.getKeysIndexes(section_name);
            int_ind = strcmp(key_names, key_name);
            ind = key_indicies(int_ind);
        end
        
        %------------------------------------------------------------------
        function updateSectionsInfo(obj)
            %updateSectionsInfo - update info about sections
            
            keys_data = obj.config_data_array(:,1);
            sect_indicies_cell = regexp(keys_data, '^\[.*\]$');
            
            obj.indicies_of_sections = ...
                find(~cellfun('isempty', sect_indicies_cell));
            
            obj.count_sections = length(obj.indicies_of_sections);
        end
        
        %------------------------------------------------------------------
        function updateCountKeysInfo(obj)
            %UpdateCountKeys - update full number of keys
            
            obj.count_all_keys = ...
                obj.count_strings - obj.count_sections - obj.count_empty_strings;
        end
        
        %------------------------------------------------------------------
        function updateEmptyStringsInfo(obj)
            %updateEmptyStringsInfo - update info about empty strings
            
            keys_data = obj.config_data_array(:,1);
            indicies_of_empty_cell = strcmp('', keys_data);
            obj.indicies_of_empty_strings = find(indicies_of_empty_cell);
            obj.count_empty_strings = length(obj.indicies_of_empty_strings);
        end
        
        %------------------------------------------------------------------
        function updateCountStrings(obj)
            %updateCountStrings - update full number of sections
            
            obj.count_strings = size(obj.config_data_array, 1);
        end
        
        %------------------------------------------------------------------
        function status = isUniqueKeyName(obj, section_name, key_name)
            %isUniqueKeyName - check whether the name of the key unique
            
            keys = obj.getKeys(section_name);
            status = ~any(strcmp(key_name, keys));
        end
        
        %------------------------------------------------------------------
        function status = isUniqueSectionName(obj, section_name)
            %isUniqueKeyName - check whether the name of the section a unique
            
            sections = obj.GetSections();
            status = ~any(strcmp(section_name, sections));
        end
        
        %------------------------------------------------------------------
        function status = isSection(obj, section_name)
            %isSection - determine whether there is a section
            
%             section_names = obj.GetSections();
            
            data = obj.config_data_array(:, 1);
            int_ind = find(strcmp(data, section_name), 1);
            
            if ~isempty(int_ind)
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function section_name = validateSectionName(obj, section_name)
            %validateSectionName - check the name of the section
            
            CheckIsString(section_name);
            
            section_name = section_name(:)';
            section_name = strtrim(section_name);
            
            if ~isempty(section_name)
                sect_indicies_cell = ...
                    regexp(section_name, '^\[.*\]$', 'once');
                
                indicies_cell_comment = ...
                    regexp(section_name, obj.comment_style, 'once');
                
                if ~isempty(indicies_cell_comment)
                    section_name = [];
                    return;
                end
                
                if isempty(sect_indicies_cell)
                    section_name = ['[', section_name, ']'];
                end
            else
                section_name = [];
            end
        end
        
        %------------------------------------------------------------------
        function status = addSection(obj, section_name)
            %addSection - add section to end configuration
            
            status = obj.insertSection(obj.count_sections+1, section_name);
        end
        
        %------------------------------------------------------------------
        function status = insertSection(obj, section_pos, section_name)
            %insertSection - insert section to given position
            
            CheckIsScalarPositiveInteger(section_pos);
            
            if (section_pos > obj.count_sections+1)
                section_pos = obj.count_sections+1;
            end
                        
            if ~isempty(section_name)
                is_unique_sect = obj.isUniqueSectionName(section_name);
                
                if ~is_unique_sect
                    status = false;
                    return;
                end
                
                if (section_pos <= obj.count_sections && obj.count_sections > 0)
                    sect_ind = obj.indicies_of_sections(section_pos);
                    
                elseif (section_pos == 1 && obj.count_sections == 0)
                    sect_ind = 1;
                    obj.config_data_array = {};
                    
                elseif (section_pos == obj.count_sections+1)
                    sect_ind = obj.count_strings+1;
                end
                
                new_data = cell(2,3);
                new_data(1,:) = {section_name, '', ''};
                new_data(2,:) = {''};
                
                obj.config_data_array = ...
                    InsertCell(obj.config_data_array, sect_ind, new_data);
                
                obj.updateCountStrings();
                obj.updateSectionsInfo();
                obj.updateEmptyStringsInfo();
                
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function status = removeSection(obj, section_name)
            %removeSection - remove given section
            
            section_name = obj.validateSectionName(section_name);
            sect_num = obj.nameToNumSection(section_name);
            
            if ~isempty(sect_num)
                if (sect_num < obj.count_sections)
                    first_ind = obj.indicies_of_sections(sect_num);
                    last_ind = obj.indicies_of_sections(sect_num+1)-1;
                    
                elseif (sect_num == obj.count_sections)
                    first_ind = obj.indicies_of_sections(sect_num);
                    last_ind = obj.count_strings;
                end
                
                obj.config_data_array(first_ind:last_ind,:) = [];
                
                obj.updateCountStrings();
                obj.updateSectionsInfo();
                obj.updateEmptyStringsInfo();
                obj.updateCountKeysInfo();
                
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function status = renameSection(obj, old_section_name, new_section_name)
            %renameSection - rename given section
            
            old_section_name = obj.validateSectionName(old_section_name);
            new_section_name = obj.validateSectionName(new_section_name);
            sect_num = obj.nameToNumSection(old_section_name);
            
            if (~isempty(new_section_name) && ~isempty(sect_num))
                sect_ind = obj.indicies_of_sections(sect_num);
                
                obj.config_data_array(sect_ind, 1) = {new_section_name};
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function key_name = validateKeyName(obj, key_name)
            %validateKeyName - check the name of the key
            
            CheckIsString(key_name);
            
            key_name = key_name(:)';
            key_name = strtrim(key_name);
            
            indicies_cell = regexp(key_name, '^\[.*\]$', 'once');
            indicies_cell_comment = regexp(key_name, obj.comment_style, 'once');
            
            if (isempty(key_name) || ~isempty(indicies_cell) || ...
                    ~isempty(indicies_cell_comment))
                
                key_name = '';
            end
        end
        
        %------------------------------------------------------------------
        function status = isKey(obj, section_name, key_name)
            %isKey - determine whether there is a key in a given section
            
            key_name = obj.validateKeyName(key_name);
            
            if ~isempty(key_name)
                status = ~obj.isUniqueKeyName(section_name, key_name);
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function [status, write_value] = ...
                addKey(obj, section_name, key_name, key_value, value_formats)
            %addKey - add key in a end given section
            
            [inds, count_keys] = obj.getKeysIndexes(section_name);
            
            [status, write_value] = obj.insertKey(section_name, ...
                count_keys+1, key_name, key_value, value_formats);
        end
        
        %------------------------------------------------------------------
        function [status, set_status] = ...
                insertKey(obj, section_name, key_pos, key_name, key_value, value_formats)
            %insertKey - insert key into the specified position in a given section
            
            CheckIsScalarPositiveInteger(key_pos);
            
            set_status = false;
            
            key_name = obj.validateKeyName(key_name);
            sect_num = obj.nameToNumSection(section_name);
            
            if (~isempty(sect_num) && ~isempty(key_name))
                is_unique_key = obj.isUniqueKeyName(section_name, key_name);
                
                if ~is_unique_key
                    status = false;
                    return;
                end
                
                [key_indicies, count_keys] = obj.getKeysIndexes(section_name);
                if (count_keys > 0)
                    if (key_pos <= count_keys)
                        insert_index = key_indicies(key_pos);
                    elseif (key_pos > count_keys)
                        insert_index = key_indicies(end) + 1;
                    end
                else
                    insert_index = obj.indicies_of_sections(sect_num) + 1;
                end
                
                new_data = {key_name, '', ''};
                
                obj.config_data_array = InsertCell(obj.config_data_array, ...
                    insert_index, new_data);
                
                obj.updateCountStrings();
                obj.updateSectionsInfo();
                obj.updateEmptyStringsInfo();
                obj.updateCountKeysInfo();
                
                if ~isempty(key_value)
                    set_status = obj.setValue(section_name, key_name, ...
                        key_value, value_formats);
                end
                
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function status = removeKey(obj, section_name, key_name)
            %removeKey - remove the key from a given section
            
            key_name = obj.validateKeyName(key_name);
            sect_num = obj.nameToNumSection(section_name);
            [keys, count_keys] = obj.getKeys(section_name);
            
            if (~isempty(sect_num) && ~isempty(key_name) && count_keys > 0)
                is_unique_key = obj.isUniqueKeyName(section_name, key_name);
                if is_unique_key
                    status = false;
                    return;
                end
                
                status = find(strcmp(key_name, keys), 1, 'last');
                key_indicies = obj.getKeysIndexes(section_name);
                
                key_index = key_indicies(status);
                obj.config_data_array(key_index, :) = [];
                
                obj.updateCountStrings();
                obj.updateSectionsInfo();
                obj.updateEmptyStringsInfo();
                obj.updateCountKeysInfo();
                
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function status = renameKey(obj, section_name, old_key_name, new_key_name)
            %renameKey - rename the key in a given section
            
            old_key_name = obj.validateKeyName(old_key_name);
            new_key_name = obj.validateKeyName(new_key_name);
            
            sect_num = obj.nameToNumSection(section_name);
            [keys, count_keys] = obj.getKeys(section_name);
            
            if (~isempty(sect_num) && ~isempty(old_key_name) && ...
                    ~isempty(new_key_name) && count_keys > 0)
                
                is_unique_key = obj.isUniqueKeyName(section_name, old_key_name);
                
                if is_unique_key
                    status = false;
                    return;
                end
                
                status = find(strcmp(old_key_name, keys), 1, 'last');
                key_indicies = obj.getKeysIndexes(section_name);
                
                key_index = key_indicies(status);
                obj.config_data_array{key_index, 1} = new_key_name;
                
                status = true;
            else
                status = false;
            end
        end
        
        %------------------------------------------------------------------
        function status = setValue(obj, section_name, key_name, key_value, value_format)
            %setValue
            
            key_name = obj.validateKeyName(key_name);
            
            if ~obj.isSection(section_name)
                status = false;
                return;
            end
            if ~obj.isKey(section_name, key_name)
                status = false;
                return;
            end
            
            key_index = obj.getKeyIndex(section_name, key_name);
            
            if isempty(value_format)
                str_value = num2str(key_value);
            else
                value_format = ValidateValueFormat(value_format);
                str_value = num2str(key_value, value_format);
            end
            
            if (isvector(key_value) && isnumeric(key_value))
                str_value = CorrectionNumericArrayStrings(str_value);
            end
            
            obj.config_data_array(key_index, 2) = {str_value};
            status = true;
        end
        
        %------------------------------------------------------------------
        function [key_value, status] = getValue(obj, ...
                section_name, key_name, default_value)
            %getValue - get key value
            
            status = false;
            
            if ~obj.isSection(section_name)
                key_value = default_value;
                return;
            end
            if ~obj.isKey(section_name, key_name)
                key_value = default_value;
                return;
            end
            
            key_index = obj.getKeyIndex(section_name, key_name);
            
            str_value = obj.config_data_array{key_index, 2};
            key_value = ParseValue(str_value);
            
            status = true;
        end
        
        %------------------------------------------------------------------
        function [key_names, count_keys] = getKeys(obj, section_name)
            %getKeys
            
            [key_indicies, count_keys] = obj.getKeysIndexes(section_name);
            
            if ~isempty(key_indicies)
                key_names = obj.config_data_array(key_indicies, 1);
            else
                key_names = {};
            end
        end
        
    end % private methods
    %----------------------------------------------------------------------
    
end % classdef IniConfig
%--------------------------------------------------------------------------


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tools Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
function [file_data, status] = GetDataFromFile(file_name)
    %GetDataFromFile - Get data from file
    
    fid = fopen(file_name, 'r');
    
    if (fid ~= -1)
        file_data = textscan(fid, ...
            '%s', ...
            'delimiter', '\n', ...
            'endOfLine', '\r\n');
        
        fclose(fid);
        
        status = true;
        file_data = file_data{1};
    else
        status = false;
        file_data = {};
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function config_data = ParseConfigData(file_data, comment_style)
    %ParseConfigData - parse data from the INI file
    
    % Select the comment in a separate array
    pat = sprintf('^[^%s]+', comment_style);
    comment_data = regexprep(file_data, pat, '');
    
    % Deleting comments
    pat = sprintf('%s.*+$', comment_style);
    file_data = regexprep(file_data, pat, '');
    
    % Select the key value in a separate array
    values_data = regexprep(file_data, '^.[^=]*.', '');
    
    % Select the names of the sections and keys in a separate array
    keys_data = regexprep(file_data, '=.*$', '');
    
    config_data = cell(size(file_data, 1), 3);
    config_data(:,1) = keys_data;
    config_data(:,2) = values_data;
    config_data(:,3) = comment_data;
    config_data = strtrim(config_data);
end
%--------------------------------------------------------------------------

%==========================================================================
function value = ParseValue(value)
    %ParseValue - classify the data types and convert them

    % определяем, содержит ли строка value не числовые символы
    start_idx = regexp(value, '[^\.\s-+,0-9ij]', 'once');
    
    if ~isempty(start_idx)
        return;
    end
    
    num = StringToNumeric(value);
    
    if ~isnan(num)
        value = num;
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function num = StringToNumeric(str)
    %StringToNumeric - convert string to numeric data
    
    if isempty(str)
        num = NaN;
        return;
    end
    
    delimiter = ',';
    
    str = regexprep(str, '\s*,*\s*', delimiter);
    cells = textscan(str, '%s', 'delimiter', delimiter);
    cells = cells{:}';
    
    num_cell = cellfun(@(x) str2double(x), cells, 'UniformOutput', false);
    is_nans = cellfun(@(x) isnan(x), num_cell, 'UniformOutput', true);
    
    if any(is_nans)
        num = NaN;
    else
        num = cell2mat(num_cell);
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function values = CorrectionNumericArrayStrings(values)
    %CorrectionNumericArrayStrings - correction strings of numeric arrays
    
    values = regexprep(values, '\s+', ', ');
end
%--------------------------------------------------------------------------

%==========================================================================
function comment_style = ValidateCommentStyle(comment_style)
    %ValidateCommentStyle - validate style of comments
    
    if ~ischar(comment_style)
        error('Requires char input for comment style.')
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function key_values = ValidateValues(key_values)
    %ValidateValues - validate data of key values
        
    is_valid = cellfun(@(x) ...
        (isempty(x) | ischar(x) | (isnumeric(x) & isvector(x))), ...
        key_values, 'UniformOutput', true);
    
    if ~all(is_valid)
        error('Invalid type of one or more <%s>.', inputname(key_values));
    end
    
    % transform key_values to vector-rows
    key_values = cellfun(@(x) x(:).', key_values, 'UniformOutput', 0);
end
%--------------------------------------------------------------------------

%==========================================================================
function value_format = ValidateValueFormat(value_format)
    %ValidateValueFormat - 
    
    CheckIsString(value_format);
    value_format = strtrim(value_format);
    
    valid_formats = 'd|i|u|f|e|g|E|G';
    
    start_ind = regexp(value_format, ...
        ['^%\d*\.?\d*(', valid_formats, ')$'], 'once');
    
    if isempty(start_ind)
        error('Invalid value format "%s".', value_format)
    end
    
    value_format = strrep(value_format, '%', '% ');
end
%--------------------------------------------------------------------------

%==========================================================================
function CheckEqualNumberElems(input1, input2)
    %CheckEqualNumberElems - checking equal numbers of elements
    
    if (numel(input1) ~= numel(input2))
        error(['Number of elements in the <%s> and ', ...
            '<%s> must be equal.'], inputname(1), inputname(2))
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function CheckIsString(input_var)
    %CheckIsString
    
    if ~(ischar(input_var) && isvector(input_var))
        error('<%s> must be a string.', inputname(1))
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function CheckIsScalarPositiveInteger(input_var)
    %CheckIsScalarPositiveInteger
    
    if ~(isscalar(input_var) && isnumeric(input_var))
        error('<%s> must be a scalar.', inputname(1))
    end
    if (input_var < 1)
        error('<%s> must be a positive integer > 0.', inputname(1))
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function cell_data = PadDataToCell(data, number_elems)
	%PadDataToCell - pad data to cell array
    
    cell_data = {data};
    cell_data = cell_data(ones(1, number_elems), :);
end
%--------------------------------------------------------------------------

%==========================================================================
function data_var = DataToCell(data_var)
    %DataToCell - convert data to cell array
    
    if ~iscell(data_var)
        data_var = {data_var};
    else
        data_var = data_var(:);
    end
end
%--------------------------------------------------------------------------

%==========================================================================
function B = InsertCell(C, i, D)
    %InsertCell - insert a new cell or several cells in a two-dimensional
    % array of cells on the index
    
    B = [C(1:i-1, :); D; C(i:end, :)];
end
%--------------------------------------------------------------------------

