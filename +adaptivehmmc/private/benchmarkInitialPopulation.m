function population = benchmarkInitialPopulation(pivots, widths, numMoments, totalNumber, volumeScale)
%BENCHMARKINITIALPOPULATION Discretize the default analytical benchmark density.

pivots = pivots(:);
widths = widths(:);
numClasses = numel(pivots);
momentOrders = 0:(numMoments - 1);
population = zeros(numClasses, 1);

for i = 1:numClasses
    lower = pivots(i) - widths(i) / 2;
    upper = pivots(i) + widths(i) / 2;
    lower = max(lower, 0);

    localMoments = zeros(numMoments, 1);
    for k = 1:numMoments
        order = momentOrders(k);
        localMoments(k) = integral(@(x) benchmarkDensity(x, totalNumber, volumeScale) .* x.^order, ...
            max(lower, realmin), upper, 'AbsTol', 1e-14, 'RelTol', 1e-10);
    end

    centerIndex = find(pivots < pivots(i), 1, 'last');
    if isempty(centerIndex)
        centerIndex = 1;
    end

    first = max(centerIndex - floor(numMoments / 2) + 1, 1);
    last = first + numMoments - 1;
    if last > numClasses
        last = numClasses;
        first = numClasses - numMoments + 1;
    end

    stencil = first:last;
    weights = solveMomentWeights(pivots(stencil), localMoments, momentOrders);
    population(stencil) = population(stencil) + weights;
end

initialNumber = sum(population);
if initialNumber > 0
    population = population * totalNumber / initialNumber;
end

end

function y = benchmarkDensity(x, totalNumber, volumeScale)
y = 3 * totalNumber / volumeScale .* x.^2 .* exp(-x.^3 / volumeScale);
end
