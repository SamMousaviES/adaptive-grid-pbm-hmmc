function tests = testAdaptiveHMMC
%TESTADAPTIVEHMMC Unit tests for the adaptivehmmc package.

tests = functiontests(localfunctions);
end

function setupOnce(testCase)
rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);
testCase.TestData.RootDir = rootDir;
end

function teardown(testCase) %#ok<INUSD>
close all force;
end

function testPackageLoads(testCase)
options = adaptivehmmc.defaultOptions();
verifyTrue(testCase, isstruct(options));
verifyEqual(testCase, options.numClasses, 20);
result = adaptivehmmc.solve(setfield(options, 'timeSpan', [0, 5])); %#ok<SFLD>
verifyTrue(testCase, isfield(result, 'moments'));
verifyGreaterThan(testCase, numel(result.time), 1);
end

function testExamplesExecute(testCase)
exampleDir = fullfile(testCase.TestData.RootDir, 'examples');
oldVisibility = get(0, 'DefaultFigureVisible');
set(0, 'DefaultFigureVisible', 'off');
cleanup = onCleanup(@() set(0, 'DefaultFigureVisible', oldVisibility));

run(fullfile(exampleDir, 'run_quickstart.m'));
run(fullfile(exampleDir, 'compare_fixed_adaptive.m'));
run(fullfile(exampleDir, 'custom_initial_distribution.m'));
verifyTrue(testCase, true);
end

function testVolumeMomentConserved(testCase)
options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 200];
options.numClasses = 16;
options.adaptive = true;

result = adaptivehmmc.solve(options);
relativeVolumeChange = abs(result.moments(4, end) - result.moments(4, 1)) / abs(result.moments(4, 1));
verifyLessThan(testCase, relativeVolumeChange, 1e-6);
end

function testAdaptiveImprovesOverNarrowFixedGrid(testCase)
options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 600];
options.numClasses = 16;
options.fMult = 1.4;

fixedOptions = options;
fixedOptions.adaptive = false;
fixedResult = adaptivehmmc.solve(fixedOptions);

adaptiveOptions = options;
adaptiveOptions.adaptive = true;
adaptiveResult = adaptivehmmc.solve(adaptiveOptions);

reference = constantKernelReference(options.timeSpan(end), fixedResult.moments(:, 1), options.process.rate);
fixedError = max(abs((fixedResult.moments(:, end) - reference) ./ reference));
adaptiveError = max(abs((adaptiveResult.moments(:, end) - reference) ./ reference));

verifyLessThan(testCase, adaptiveError, fixedError);
end

function testRedistributionPreservesMoments(testCase)
options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 200];
options.numClasses = 16;
options.adaptive = true;

result = adaptivehmmc.solve(options);
duplicateIndex = find(diff(result.time) == 0, 1, 'first');
verifyNotEmpty(testCase, duplicateIndex);

before = result.moments(:, duplicateIndex);
after = result.moments(:, duplicateIndex + 1);
relativeDifference = max(abs(after - before) ./ max(abs(before), eps));
verifyLessThan(testCase, relativeDifference, 1e-5);
end

function testInvalidOptionsError(testCase)
options = adaptivehmmc.defaultOptions();
options.numClasses = 3;
options.numMoments = 6;
verifyError(testCase, @() adaptivehmmc.solve(options), 'adaptivehmmc:InvalidOptions');
end

function testSolverDoesNotOpenFigures(testCase)
close all force;
before = numel(findall(0, 'Type', 'figure'));
options = adaptivehmmc.defaultOptions();
options.timeSpan = [0, 20];
options.numClasses = 10;
adaptivehmmc.solve(options);
after = numel(findall(0, 'Type', 'figure'));
verifyEqual(testCase, after, before);
end

function reference = constantKernelReference(time, initialMoments, rate)
initialNumber = initialMoments(1);
factor = 2 / (2 + rate * initialNumber * time);
reference = zeros(size(initialMoments));
for k = 1:numel(initialMoments)
    order = k - 1;
    reference(k) = initialMoments(k) * factor^(1 - order / 3);
end
end
