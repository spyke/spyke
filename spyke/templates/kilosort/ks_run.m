% adapted from KiloSort's 'master_file_example_MOVEME.m' template

addpath(genpath('~/src/KiloSort')) % path to KiloSort folder
addpath(genpath('~/src/npy-matlab')) % path to npy-matlab scripts

run('{KSCONFIGFNAME}')

tic; % start timer

if ops.GPU     
    gpuDevice(1); % initialize GPU (will erase any existing GPU arrays)
end

if strcmp(ops.datatype, 'openEphys')
   ops = convertOpenEphysToRawBInary(ops); % convert data, only for OpenEphys
end

[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez = fitTemplates(rez, DATA, uproj); % fit templates iteratively
rez = fullMPMU(rez, DATA); % extract final spike times (overlapping extraction)

% AutoMerge. rez2Phy will use for clusters the new 5th column of st3 if you run this
%rez = merge_posthoc2(rez);

% save matlab results file
if ~isdir(ops.root)
    mkdir('.', ops.root) % create save folder if it doesn't exist
end
save(fullfile(ops.root, 'rez.mat'), 'rez', '-v7.3');

% save python results file for Phy
rezToPhy(rez, ops.root);

% remove temporary file
delete(ops.fproc);
