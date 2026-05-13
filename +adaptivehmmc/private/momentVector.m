function moments = momentVector(pivots, population, numMoments)
%MOMENTVECTOR Calculate diameter moments M_0 ... M_(numMoments-1).

pivots = pivots(:);
population = population(:);
moments = zeros(numMoments, 1);

for k = 1:numMoments
    moments(k) = sum(population .* pivots.^(k - 1));
end

end
