%COMPARE_FIXED_ADAPTIVE Compare fixed and adaptive grids at equal class count.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);

baseOptions = adaptivehmmc.defaultOptions();
baseOptions.timeSpan = [0, 1000];
baseOptions.numClasses = 20;
baseOptions.fMult = 1.4;

fixedOptions = baseOptions;
fixedOptions.adaptive = false;
fixedResult = adaptivehmmc.solve(fixedOptions);

adaptiveOptions = baseOptions;
adaptiveOptions.adaptive = true;
adaptiveResult = adaptivehmmc.solve(adaptiveOptions);

reference = constantKernelReference(baseOptions.timeSpan(end), ...
    fixedResult.moments(:, 1), baseOptions.process.rate);

fixedError = max(abs((fixedResult.moments(:, end) - reference) ./ reference)) * 100;
adaptiveError = max(abs((adaptiveResult.moments(:, end) - reference) ./ reference)) * 100;

fprintf('Fixed-grid max moment error: %.3f %%\n', fixedError);
fprintf('Adaptive-grid max moment error: %.3f %%\n', adaptiveError);

figure('Color', 'w');
semilogy(fixedResult.time, momentErrorHistory(fixedResult, reference, baseOptions.timeSpan(end)), ...
    'k--', 'LineWidth', 1.5);
hold on;
semilogy(adaptiveResult.time, momentErrorHistory(adaptiveResult, reference, baseOptions.timeSpan(end)), ...
    'k-', 'LineWidth', 1.5);
grid on;
box on;
xlabel('Time');
ylabel('Max. moment error [%]');
legend({'Fixed grid', 'Adaptive grid'}, 'Location', 'northwest');
title('Fixed versus adaptive grid');

function reference = constantKernelReference(time, initialMoments, rate)
initialNumber = initialMoments(1);
factor = 2 / (2 + rate * initialNumber * time);
reference = zeros(size(initialMoments));
for k = 1:numel(initialMoments)
    order = k - 1;
    reference(k) = initialMoments(k) * factor^(1 - order / 3);
end
end

function errors = momentErrorHistory(result, finalReference, finalTime)
errors = zeros(size(result.time));
initialMoments = result.moments(:, 1);
rate = result.options.process.rate;
for i = 1:numel(result.time)
    reference = constantKernelReference(result.time(i), initialMoments, rate);
    if finalTime == result.time(i)
        reference = finalReference;
    end
    errors(i) = max(abs((result.moments(:, i) - reference) ./ reference)) * 100;
end
end
