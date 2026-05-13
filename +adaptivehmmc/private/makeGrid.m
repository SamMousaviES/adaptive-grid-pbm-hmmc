function [pivots, widths] = makeGrid(numClasses, diameterMin, diameterMax, gridRatio)
%MAKEGRID Create a monotone pivot grid with geometric interval widths.

if gridRatio == 1
    weights = ones(numClasses, 1);
else
    weights = gridRatio .^ (0:(numClasses - 1));
    weights = weights(:);
end

widths = weights / sum(weights) * (diameterMax - diameterMin);
edges = diameterMin + [0; cumsum(widths)];
pivots = 0.5 * (edges(1:end - 1) + edges(2:end));

end
