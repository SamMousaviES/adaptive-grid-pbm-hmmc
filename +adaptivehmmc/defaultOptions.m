function options = defaultOptions()
%DEFAULTOPTIONS Return default options for adaptivehmmc.solve.
%
%   options = adaptivehmmc.defaultOptions() returns a struct suitable for
%   the constant-kernel coalescence benchmark. Edit fields in the returned
%   struct and pass it to adaptivehmmc.solve.

options = struct();

options.timeSpan = [0, 1000];
options.numClasses = 20;
options.numMoments = 6;

options.diameterMin = 0;
options.diameterMax = 1;
options.gridRatio = 1.3;

options.adaptive = true;
options.fMax = 2.5;
options.fMult = 1.4;

options.initialDistribution = struct();
options.initialDistribution.type = 'benchmark';
options.initialDistribution.totalNumber = 1;
options.initialDistribution.volumeScale = 1.0e-3;
options.initialDistribution.pivots = [];
options.initialDistribution.population = [];

options.process = struct();
options.process.type = 'coalescence';
options.process.kernel = 'constant';
options.process.rate = 1;

options.solver = struct();
options.solver.RelTol = 1.0e-6;
options.solver.AbsTol = 1.0e-9;
options.solver.MaxStep = [];
options.solver.storeHistory = true;

end
