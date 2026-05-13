function options = validateOptions(options)
%VALIDATEOPTIONS Validate and normalize user options.

mustBeNumericVector(options.timeSpan, 'timeSpan');
if numel(options.timeSpan) < 2 || options.timeSpan(1) < 0 || options.timeSpan(end) <= options.timeSpan(1)
    error('adaptivehmmc:InvalidOptions', 'timeSpan must contain at least two increasing nonnegative times.');
end

options.numClasses = validateInteger(options.numClasses, 'numClasses', 2, Inf);
options.numMoments = validateInteger(options.numMoments, 'numMoments', 1, options.numClasses);

mustBeFiniteScalar(options.diameterMin, 'diameterMin');
mustBeFiniteScalar(options.diameterMax, 'diameterMax');
if options.diameterMin < 0 || options.diameterMax <= options.diameterMin
    error('adaptivehmmc:InvalidOptions', 'diameterMax must be greater than nonnegative diameterMin.');
end

mustBeFiniteScalar(options.gridRatio, 'gridRatio');
if options.gridRatio < 1
    error('adaptivehmmc:InvalidOptions', 'gridRatio must be at least 1.');
end

if ~(islogical(options.adaptive) || isnumeric(options.adaptive))
    error('adaptivehmmc:InvalidOptions', 'adaptive must be true or false.');
end
options.adaptive = logical(options.adaptive);

mustBeFiniteScalar(options.fMax, 'fMax');
mustBeFiniteScalar(options.fMult, 'fMult');
if options.fMax <= 0 || options.fMult <= 1
    error('adaptivehmmc:InvalidOptions', 'fMax must be positive and fMult must be greater than 1.');
end

if ~isstruct(options.initialDistribution)
    error('adaptivehmmc:InvalidOptions', 'initialDistribution must be a struct.');
end

if ~isfield(options.initialDistribution, 'type') || ~ischarOrString(options.initialDistribution.type)
    error('adaptivehmmc:InvalidOptions', 'initialDistribution.type must be a character vector or string.');
end
options.initialDistribution.type = lower(char(options.initialDistribution.type));

if ~isfield(options.initialDistribution, 'pivots')
    options.initialDistribution.pivots = [];
end
if ~isfield(options.initialDistribution, 'population')
    options.initialDistribution.population = [];
end

if strcmp(options.initialDistribution.type, 'custom')
    if isempty(options.initialDistribution.population)
        error('adaptivehmmc:InvalidOptions', 'Custom initialDistribution requires a population vector.');
    end
    if ~isempty(options.initialDistribution.pivots)
        if numel(options.initialDistribution.pivots) ~= numel(options.initialDistribution.population)
            error('adaptivehmmc:InvalidOptions', 'Custom pivots and population must have the same length.');
        end
        options.numClasses = numel(options.initialDistribution.population);
        if options.numMoments > options.numClasses
            error('adaptivehmmc:InvalidOptions', 'numMoments cannot exceed the custom population length.');
        end
    elseif numel(options.initialDistribution.population) ~= options.numClasses
        error('adaptivehmmc:InvalidOptions', 'Custom population length must match numClasses.');
    end
end

if ~isstruct(options.process)
    error('adaptivehmmc:InvalidOptions', 'process must be a struct.');
end
options.process.type = lower(char(options.process.type));
options.process.kernel = lower(char(options.process.kernel));
if ~strcmp(options.process.type, 'coalescence') || ~strcmp(options.process.kernel, 'constant')
    error('adaptivehmmc:UnsupportedProcess', 'Only constant-kernel coalescence is supported in this release.');
end
mustBeFiniteScalar(options.process.rate, 'process.rate');
if options.process.rate <= 0
    error('adaptivehmmc:InvalidOptions', 'process.rate must be positive.');
end

mustBeFiniteScalar(options.solver.RelTol, 'solver.RelTol');
mustBeFiniteScalar(options.solver.AbsTol, 'solver.AbsTol');
if options.solver.RelTol <= 0 || options.solver.AbsTol <= 0
    error('adaptivehmmc:InvalidOptions', 'Solver tolerances must be positive.');
end

if ~isempty(options.solver.MaxStep)
    mustBeFiniteScalar(options.solver.MaxStep, 'solver.MaxStep');
    if options.solver.MaxStep <= 0
        error('adaptivehmmc:InvalidOptions', 'solver.MaxStep must be positive when specified.');
    end
end

if ~(islogical(options.solver.storeHistory) || isnumeric(options.solver.storeHistory))
    error('adaptivehmmc:InvalidOptions', 'solver.storeHistory must be true or false.');
end
options.solver.storeHistory = logical(options.solver.storeHistory);

end

function value = validateInteger(value, name, lowerBound, upperBound)
mustBeFiniteScalar(value, name);
if value ~= round(value) || value < lowerBound || value > upperBound
    error('adaptivehmmc:InvalidOptions', '%s must be an integer in the allowed range.', name);
end
value = double(value);
end

function mustBeFiniteScalar(value, name)
if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
    error('adaptivehmmc:InvalidOptions', '%s must be a finite numeric scalar.', name);
end
end

function mustBeNumericVector(value, name)
if ~isnumeric(value) || ~isvector(value) || any(~isfinite(value))
    error('adaptivehmmc:InvalidOptions', '%s must be a finite numeric vector.', name);
end
end

function tf = ischarOrString(value)
tf = ischar(value) || (isstring(value) && isscalar(value));
end
