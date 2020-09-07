% file names and paths:
%ops.datatype = 'dat';  % binary ('dat', 'bin') or 'openEphys'
ops.fbinary = '{DATFNAME}';
% save results to this local folder, created if missing
ops.root = '{KSRESULTSFOLDERNAME}';
% channel map (either a string filename, or an array for linear probe: 1:ops.Nchan)
ops.chanMap = '{CHANMAPFNAME}';
% residual from RAM of preprocessed data
ops.fproc = 'temp_wh.dat';

% time range and channel count:
ops.trange = [0 Inf]; % time range to sort
ops.NchanTOT = {NCHANS}; % total number of channels (omit if already in chanMap file)
%ops.Nchan = {NCHANS}; % number of active channels (omit if already in chanMap file)

% sample rate (30000)
ops.fs = {FS};

% frequency for high pass filtering (150)
ops.fshigh = 300;

% minimum firing rate on a "good" channel (0.1, 0 to skip)
ops.minfr_goodchannels = 0.05;

% threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
ops.Th = [10 4];

% how important is the amplitude penalty
% (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
ops.lam = 10;

% splitting a cluster at the end requires at least this much isolation for each sub-cluster
% (max = 1)
ops.AUCsplit = 0.9;

% minimum spike rate (Hz), if a cluster falls below this for too long it gets removed (1/50)
ops.minFR = 0.01;

% number of samples to average over (annealed from first to second value) ([20 400])
ops.momentum = [20 400];

% spatial constant in um for computing residual variance of spike (30)
ops.sigmaMask = 30;

% threshold crossings for pre-clustering (in PCA projection space) (8)
ops.ThPre = 8;

%% danger, changing these settings can lead to fatal errors:

% options for determining PCs
ops.spkTh          = -6; % spike threshold in standard deviations (-6)
ops.reorder        = 1; % whether to reorder batches for drift correction.
ops.nskip          = 25; % how many batches to skip for determining spike PCs

ops.GPU            = 1; % has to be 1, no CPU version yet, sorry
%ops.Nfilt          = 1024; % max number of clusters
ops.nfilt_factor   = 4; % max number of clusters per good channel (even temporary ones)
ops.ntbuff         = 64; % samples of symmetrical buffer for whitening and spike detection
ops.NT             = 64*1024 + ops.ntbuff; % must be multiple of 32 + ntbuff. This is the
                                           % batch size (try decreasing if out of memory).
ops.whiteningRange = 32; % number of channels to use for whitening each channel
ops.nSkipCov       = 25; % compute whitening matrix from every N-th batch
ops.scaleproc      = 200;   % int16 scaling of whitened data
ops.nPCs           = 3; % how many PCs to project the spikes into
ops.useRAM         = 0; % not yet available

%%
