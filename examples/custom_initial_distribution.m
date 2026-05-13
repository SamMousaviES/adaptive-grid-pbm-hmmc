%CUSTOM_INITIAL_DISTRIBUTION Solve with user-defined pivots and populations.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);

pivots = linspace(0.05, 1.0, 16);
population = exp(-0.5 * ((pivots - 0.25) / 0.08).^2);
population = population / sum(population);

options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 200];
options.numMoments = 6;
options.adaptive = true;
options.initialDistribution.type = 'custom';
options.initialDistribution.pivots = pivots;
options.initialDistribution.population = population;

result = adaptivehmmc.solve(options);

fprintf('Custom case final largest pivot: %.6g\n', result.pivots(end, end));

figure('Color', 'w');
stem(result.pivots(:, 1), result.population(:, 1), 'k:', 'filled');
hold on;
stem(result.pivots(:, end), result.population(:, end), 'k-', 'filled');
grid on;
box on;
xlabel('Diameter');
ylabel('Class population');
legend({'Initial', 'Final'}, 'Location', 'northeast');
title('Custom initial distribution');
