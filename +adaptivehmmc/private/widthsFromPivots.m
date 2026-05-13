function widths = widthsFromPivots(pivots)
%WIDTHSFROMPIVOTS Estimate interval widths from ordered pivots.

pivots = pivots(:);
numClasses = numel(pivots);
widths = zeros(numClasses, 1);

if numClasses == 1
    widths(1) = max(pivots(1), eps);
    return
end

widths(1) = max(2 * pivots(1), eps);
for i = 2:numClasses
    leftEdge = pivots(i - 1) + widths(i - 1) / 2;
    widths(i) = max(2 * (pivots(i) - leftEdge), eps);
end

end
