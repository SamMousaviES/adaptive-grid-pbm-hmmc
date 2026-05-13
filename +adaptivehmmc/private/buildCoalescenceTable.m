function table = buildCoalescenceTable(pivots, momentOrders)
%BUILDCOALESCENCETABLE Precompute product redistribution weights.

pivots = pivots(:);
numClasses = numel(pivots);
stencilSize = numel(momentOrders);

weights = zeros(numClasses, numClasses, stencilSize);
firstIndex = ones(numClasses, numClasses);

for i = 1:numClasses
    for j = i:numClasses
        productPivot = (pivots(i)^3 + pivots(j)^3)^(1 / 3);
        first = localStencilStart(pivots, productPivot, stencilSize);
        stencil = first:(first + stencilSize - 1);

        productMoments = zeros(stencilSize, 1);
        for k = 1:stencilSize
            productMoments(k) = productPivot ^ momentOrders(k);
        end

        localWeights = solveMomentWeights(pivots(stencil), productMoments, momentOrders);

        volume = sum(localWeights(:) .* pivots(stencil).^3);
        if volume ~= 0
            localWeights = localWeights * ((pivots(i)^3 + pivots(j)^3) / volume);
        end

        weights(i, j, :) = localWeights;
        weights(j, i, :) = localWeights;
        firstIndex(i, j) = first;
        firstIndex(j, i) = first;
    end
end

table = struct();
table.weights = weights;
table.firstIndex = firstIndex;
table.stencilSize = stencilSize;
table.numClasses = numClasses;

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
