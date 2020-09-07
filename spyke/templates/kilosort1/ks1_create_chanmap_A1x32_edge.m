% create a channel map file for KiloSort

Nchannels = 32;
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;
xcoords   = zeros(Nchannels,1);
ycoords   = [620-620
             620-600
             620-580
             620-560
             620-540
             620-520
             620-500
             620-480
             620-460
             620-440
             620-420
             620-400
             620-380
             620-360
             620-340
             620-320
             620-300
             620-280
             620-260
             620-240
             620-220
             620-200
             620-180
             620-160
             620-140
             620-120
             620-100
             620-80
             620-60
             620-40
             620-20
             620-0]
kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)
                               % mspacek: does this have something to do with shanks, or is
                               % that what 'connected' is for?

fs = 30000; % sampling frequency
save('ks_A1x32_chanmap_edge.mat', ...
     'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')

% kcoords is used to forcefully restrict templates to channels in the same
% channel group. An option can be set in the master_file to allow a fraction 
% of all templates to span more channel groups, so that they can capture shared 
% noise across all channels. This option is

% ops.criterionNoiseChannels = 0.2; 

% if this number is less than 1, it will be treated as a fraction of the total number of clusters

% if this number is larger than 1, it will be treated as the "effective
% number" of channel groups at which to set the threshold. So if a template
% occupies more than this many channel groups, it will not be restricted to
% a single channel group. 
