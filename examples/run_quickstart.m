%RUN_QUICKSTART Basic adaptive-grid coalescence example.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);

options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 300];
options.numClasses = 20;
options.adaptive = true;

result = adaptivehmmc.solve(options);

fprintf('Final d43: %.6g\n', result.d43(end));
fprintf('Adaptation events: %d\n', result.report.numAdaptations);

figure('Color', 'w');
plot(result.time, result.d43, 'k-', 'LineWidth', 1.5);
hold on;
plot(result.time, result.pivots(end, :), 'k--', 'LineWidth', 1.2);
grid on;
box on;
xlabel('Time');
ylabel('Diameter');
legend({'d43', 'largest pivot'}, 'Location', 'northwest');
title('Adaptive HMMC quickstart');
