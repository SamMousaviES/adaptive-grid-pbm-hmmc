function weights = solveMomentWeights(pivots, moments, momentOrders)
%SOLVEMOMENTWEIGHTS Solve local moment matching system.

pivots = pivots(:);
moments = moments(:);
momentOrders = momentOrders(:);

numWeights = numel(pivots);
matrix = zeros(numWeights, numWeights);
for row = 1:numWeights
    matrix(row, :) = pivots(:).' .^ momentOrders(row);
end

weights = matrix \ moments(1:numWeights);

end
