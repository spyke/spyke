% create a channel map file for KiloSort

Nchannels = 32;
connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;
xcoords   = zeros(Nchannels,1);
ycoords   = [775-775
             775-725
             775-675
             775-625
             775-575
             775-525
             775-475
             775-425
             775-375
             775-325
             775-275
             775-225
             775-175
             775-125
             775-75
             775-25
             775-0
             775-50
             775-100
             775-150
             775-200
             775-250
             775-300
             775-350
             775-400
             775-450
             775-500
             775-550
             775-600
             775-650
             775-700
             775-750]
kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)
                               % mspacek: does this have something to do with shanks, or is
                               % that what 'connected' is for?

fs = 30000; % sampling frequency
save('ks_A1x32_chanmap.mat', ...
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
