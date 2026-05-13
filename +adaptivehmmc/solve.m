function result = solve(options)
%SOLVE Solve a moment-conserving population balance with optional adaptation.
%
%   result = adaptivehmmc.solve(options) solves a constant-kernel
%   coalescence population balance using a fixed or uniformly scaled
%   adaptive pivot grid. The input options struct can be obtained from
%   adaptivehmmc.defaultOptions.
%
%   The returned result contains time histories for population, pivots,
%   moments, d43, target upper pivot, adaptation times, a report struct, and
%   the validated options.

if nargin < 1 || isempty(options)
    options = adaptivehmmc.defaultOptions();
else
    options = mergeStructs(adaptivehmmc.defaultOptions(), options);
end

options = validateOptions(options);

timeSpan = options.timeSpan(:).';
t0 = timeSpan(1);
tf = timeSpan(end);

[pivots, widths, population] = initializeState(options);
population = population(:);

momentOrders = 0:(options.numMoments - 1);
table = buildCoalescenceTable(pivots, momentOrders);
kernelRate = options.process.rate;

adaptationTimes = zeros(0, 1);
totalSteps = 0;
ticHandle = tic;

history.time = zeros(0, 1);
history.population = zeros(options.numClasses, 0);
history.pivots = zeros(options.numClasses, 0);
history.moments = zeros(options.numMoments, 0);
history.d43 = zeros(0, 1);
history.targetUpperPivot = zeros(0, 1);

history = appendState(history, t0, population, pivots, options.numMoments, options.fMax);

if tf > t0
    currentTime = t0;
    currentPopulation = population;
    currentPivots = pivots;
    currentWidths = widths;

    while currentTime < tf
        odeOptions = odeset( ...
            'RelTol', options.solver.RelTol, ...
            'AbsTol', options.solver.AbsTol);

        if ~isempty(options.solver.MaxStep)
            odeOptions = odeset(odeOptions, 'MaxStep', options.solver.MaxStep);
        end

        if options.adaptive
            segmentStart = currentTime;
            odeOptions = odeset(odeOptions, 'Events', ...
                @(t, y) adaptationEvent(t, y, currentPivots, options, segmentStart));
        end

        rhs = @(t, y) coalescenceRhs(t, y, table, kernelRate);
        [tSegment, ySegment] = ode15s(rhs, [currentTime, tf], currentPopulation, odeOptions);

        totalSteps = totalSteps + numel(tSegment);

        if numel(tSegment) > 1
            startIndex = 2;
            if numel(history.time) == 0
                startIndex = 1;
            end

            if options.solver.storeHistory
                for k = startIndex:numel(tSegment)
                    history = appendState(history, tSegment(k), ySegment(k, :).', ...
                        currentPivots, options.numMoments, options.fMax);
                end
            else
                history = appendState(history, tSegment(end), ySegment(end, :).', ...
                    currentPivots, options.numMoments, options.fMax);
            end
        end

        currentTime = tSegment(end);
        currentPopulation = ySegment(end, :).';

        if ~options.adaptive || currentTime >= tf
            break
        end

        [currentPopulation, currentPivots, currentWidths, wasAdapted] = ...
            redistributePopulation(currentPopulation, currentPivots, currentWidths, ...
            momentOrders, options.fMax);

        if ~wasAdapted
            break
        end

        adaptationTimes(end + 1, 1) = currentTime; %#ok<AGROW>
        history = appendState(history, currentTime, currentPopulation, ...
            currentPivots, options.numMoments, options.fMax);
    end
end

result = struct();
result.time = history.time;
result.population = history.population;
result.pivots = history.pivots;
result.moments = history.moments;
result.d43 = history.d43;
result.targetUpperPivot = history.targetUpperPivot;
result.adaptationTimes = adaptationTimes;
result.options = options;

result.report = struct();
result.report.finalTime = result.time(end);
result.report.numClasses = options.numClasses;
result.report.numMoments = options.numMoments;
result.report.numAdaptations = numel(adaptationTimes);
result.report.totalOdeSteps = totalSteps;
result.report.cpuTime = toc(ticHandle);
result.report.initialLargestPivot = history.pivots(end, 1);
result.report.finalLargestPivot = history.pivots(end, end);
result.report.finalMoments = history.moments(:, end);

end

function history = appendState(history, time, population, pivots, numMoments, fMax)
moments = momentVector(pivots, population, max(numMoments, 5));
d43Value = d43FromMoments(moments);

history.time(end + 1, 1) = time;
history.population(:, end + 1) = population(:);
history.pivots(:, end + 1) = pivots(:);
history.moments(:, end + 1) = moments(1:numMoments);
history.d43(end + 1, 1) = d43Value;
history.targetUpperPivot(end + 1, 1) = fMax * d43Value;
end

function [value, isterminal, direction] = adaptationEvent(t, y, pivots, options, segmentStart)
isterminal = 1;
direction = 0;
value = -1;

if t - segmentStart <= 1.0e-9
    return
end

moments = momentVector(pivots, y, 5);
d43Value = d43FromMoments(moments);
targetUpperPivot = options.fMax * d43Value;
largestPivot = pivots(end);

if targetUpperPivot > options.fMult * largestPivot || ...
        targetUpperPivot < largestPivot / options.fMult
    value = 0;
end
end
