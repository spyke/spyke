% add paths:
addpath(genpath('~/src/KiloSort2')) % path to KiloSort folder
addpath(genpath('~/src/npy-matlab')) % path to npy-matlab scripts

% run config file:
run('{KSCONFIGFNAME}')

% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

% time-reordering as a function of drift
rez = clusterSingleBatches(rez);

% saving results here is a good idea, because the rest can be resumed after loading rez:
%save(fullfile(ops.root, 'rez.mat'), 'rez', '-v7.3');

% main tracking and template matching algorithm
rez = learnAndSolve8b(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
rez = splitAllClusters(rez, 1);

% final splits by amplitudes
rez = splitAllClusters(rez, 0);

% decide on cutoff
rez = set_cutoff(rez);

fprintf('Found %d good units\n', sum(rez.good > 0))

% create save folder if it doesn't exist
if ~isdir(ops.root)
    mkdir('.', ops.root)
end

% write Python files
fprintf('Saving results to .npy\n')
rezToPhy(rez, ops.root);

% remove temporary file
delete(ops.fproc);

%% if you want to save the results to a Matlab file...

% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% final time sorting of spikes, for apps that use st3 directly
[~, isort] = sortrows(rez.st3);
rez.st3    = rez.st3(isort, :);

% Ensure all GPU arrays are transferred to CPU side before saving to .mat
rez_fields = fieldnames(rez);
for i = 1:numel(rez_fields)
    field_name = rez_fields{i};
    if(isa(rez.(field_name), 'gpuArray'))
        rez.(field_name) = gather(rez.(field_name));
    end
end

% save final results to rez
fprintf('Saving final results in rez\n')
fname = fullfile(ops.root, 'rez.mat');
save(fname, 'rez', '-v7.3');
