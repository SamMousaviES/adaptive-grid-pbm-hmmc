function out = mergeStructs(defaults, overrides)
%MERGESTRUCTS Recursively merge scalar option structs.

out = defaults;

if isempty(overrides)
    return
end

if ~isstruct(overrides)
    error('adaptivehmmc:InvalidOptions', 'Options must be a struct.');
end

names = fieldnames(overrides);
for i = 1:numel(names)
    name = names{i};
    if isfield(out, name) && isstruct(out.(name)) && isstruct(overrides.(name))
        out.(name) = mergeStructs(out.(name), overrides.(name));
    else
        out.(name) = overrides.(name);
    end
end

end
