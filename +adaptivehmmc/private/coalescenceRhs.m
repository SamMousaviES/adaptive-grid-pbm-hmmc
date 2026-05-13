function dydt = coalescenceRhs(~, y, table, kernelRate)
%COALESCENCERHS Constant-kernel coalescence right-hand side.

y = y(:);
numClasses = table.numClasses;
dydt = zeros(numClasses, 1);

for i = 1:numClasses
    yi = y(i);
    for j = i:numClasses
        yj = y(j);
        rate = kernelRate * yi * yj;
        if i == j
            eventRate = 0.5 * rate;
        else
            eventRate = rate;
        end

        first = table.firstIndex(i, j);
        for k = 1:table.stencilSize
            destination = first + k - 1;
            if destination <= numClasses
                dydt(destination) = dydt(destination) + eventRate * table.weights(i, j, k);
            end
        end

        dydt(i) = dydt(i) - eventRate;
        dydt(j) = dydt(j) - eventRate;
    end
end

end
