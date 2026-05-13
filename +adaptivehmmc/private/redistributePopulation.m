function [population, pivots, widths, wasAdapted] = redistributePopulation(population0, pivots0, widths0, momentOrders, fMax)
%REDISTRIBUTEPOPULATION Uniformly scale grid and project population onto it.

diagnosticMoments = momentVector(pivots0, population0, 5);
d43Value = d43FromMoments(diagnosticMoments);
targetUpperPivot = fMax * d43Value;

if ~isfinite(targetUpperPivot) || targetUpperPivot <= 0
    population = population0(:);
    pivots = pivots0(:);
    widths = widths0(:);
    wasAdapted = false;
    return
end

scaleFactor = targetUpperPivot / pivots0(end);
pivots = pivots0(:) * scaleFactor;
widths = widths0(:) * scaleFactor;
population = zeros(size(population0(:)));
stencilSize = numel(momentOrders);

for i = 1:numel(pivots0)
    sourcePivot = pivots0(i);
    sourcePopulation = population0(i);
    first = localStencilStart(pivots, sourcePivot, stencilSize);
    stencil = first:(first + stencilSize - 1);

    sourceMoments = zeros(stencilSize, 1);
    for k = 1:stencilSize
        sourceMoments(k) = sourcePivot ^ momentOrders(k);
    end

    weights = solveMomentWeights(pivots(stencil), sourceMoments, momentOrders);
    population(stencil) = population(stencil) + sourcePopulation * weights;
end

wasAdapted = true;

end

function first = localStencilStart(pivots, target, stencilSize)
numClasses = numel(pivots);
left = find(pivots < target, 1, 'last');
if isempty(left)
    left = 1;
end

first = max(left - floor(stencilSize / 2) + 1, 1);
last = first + stencilSize - 1;
if last > numClasses
    first = numClasses - stencilSize + 1;
end
end
