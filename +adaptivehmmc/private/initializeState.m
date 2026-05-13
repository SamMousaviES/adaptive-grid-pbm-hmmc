function [pivots, widths, population] = initializeState(options)
%INITIALIZESTATE Build initial grid and population.

initial = options.initialDistribution;

if strcmp(initial.type, 'custom') && ~isempty(initial.pivots)
    pivots = initial.pivots(:);
    population = initial.population(:);
    widths = widthsFromPivots(pivots);
    return
end

[pivots, widths] = makeGrid(options.numClasses, options.diameterMin, ...
    options.diameterMax, options.gridRatio);

if strcmp(initial.type, 'custom')
    population = initial.population(:);
elseif strcmp(initial.type, 'benchmark')
    population = benchmarkInitialPopulation(pivots, widths, options.numMoments, ...
        initial.totalNumber, initial.volumeScale);
else
    error('adaptivehmmc:InvalidOptions', 'Unsupported initialDistribution.type: %s.', initial.type);
end

end
