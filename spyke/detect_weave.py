"""Spike detection loops implemented in weave.inline C for speed
Using Cython instead. This is here just for reference"""

from __future__ import division

__author__ = 'Martin Spacek'

import numpy as np
import time
import wx


class BipolarAmplitudeFixedThresh_Weave(object):

    def searchblock_weave(self, wave):
        """Search across all chans in a manageable block of waveform
        data and return a tuple of spike time and maxchan arrays.
        Apply both temporal and spatial lockouts

        TODO: is there an abs f'n in C I can use? what's the most optimized way of taking abs?
        TODO: try creating xthresh, last, and lock within C loop, to reduce number of vars and amount of
              data that blitz needs to convert. Tried this before, but wasn't specifically measuring C loop speed
              Also, should retry many of the optimizations, now that I'm directly measuring C loop speed:
                - use absdata(chanii, ti) everywhere instead of copying to v
        """
        chans = np.asarray(self.chans)
        nchans = len(self.chans)
        if self.chans != range(nchans): # if self.chans is non contiguous
            ttake = time.clock()
            data = wave.data.take(chans, axis=0) # pull our chans of data out, this assumes wave.data has all possible chans in it
            print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # avoid an unnecessary .take to save time
            data = wave.data
        tabs = time.clock()
        absdata = np.abs(data) # TODO: this step takes about .03 or .04 sec for 1 sec data, which is almost as
                               # slow as the whole C loop, try doing it in C instead
        print 'abs took %.3f sec' % (time.clock()-tabs)
        nt = wave.data.shape[1]
        totalnspikes = self.totalnspikes # total num of spikes found so far in this Detector.search()
        maxnspikes = self.maxnspikes
        thresh = self.fixedthresh

        xthresh = np.zeros(nchans, dtype=np.int32) # thresh xing flags
        last = np.zeros(nchans) # holds last signal value per chan, floats in uV
        lock = np.zeros(nchans, dtype=np.int32) # holds number of lockout timepoints left per chan

        tilock = self.tilock # temporal lockout, in num timepoints
        slock = self.slock # spatial lockout, in um
        dm = self.dm # Euclidean channel distance matrix
        spiketis = self.spiketis # init'd in self.search()

        """
        TODO: after searching for maxchan, search again for peak on that maxchan,
              maybe iterate a few times until stop seeing changes in result for
              that spike

        TODO: when searching for maxchan, limit search to chans within spatial lockout.
              This will be faster, and will also prevent unrelated distant cells firing
              at the same time from triggering each other, or something

        TODO: instead of searching forward indefinitely for a peak, maybe search
              forward and backwards say 0.5ms. Ah, but then searching backwards would
              ignore any tlock you may have had on that chan in the past. Maybe that means
              having to change how tlock is implemented

        TODO: take weighted spatial average of say the top 3 channels at peak time of the
              maxchan, and use that point as the center of the spatial lockout

        TODO: add option to search backwards in time:
                - maybe just slice data in reverse order, C loop won't know the difference?
                - would also have to reverse order of block loop, can prolly do this by just
                  simply blocksize BS -ve

        """

        """
        abs f'n in C:

                for ( int ti=0; ti<nt; ti++ ) { // iterate over all timepoints
                    for ( int chanii=0; chanii<nchans; chanii++ ) { // iterate over indices into chans and data
                        if ( absdata(chanii, ti) < 0 )
                            absdata(chanii, ti) *= -1; // this is very dangerous - unless we do a copy in python, we'll be overwriting actual data
                    }
                }

        """


        code = r"""
        #line 295 "detect.py" // for debugging
        int spikei, maxchanii, maxchani, chanj;
        double v; // current signal voltage, uV (Python float), using a pointer doesn't seem faster
        for ( int ti=0; ti<nt; ti++ ) { // iterate over all timepoints
            // TODO: chan loop should go in random order on each ti, to prevent a chan from
            // dominating with its spatial lockout or something like that
            for ( int chanii=0; chanii<nchans; chanii++ ) { // iterate over indices into chans
                if ( lock(chanii) > 0 ) // if this chan is still locked out
                    lock(chanii)--; // decr this chan's temporal lockout
                else { // search for a thresh xing or a peak
                    v = absdata(chanii, ti);
                    if ( xthresh(chanii) == 0 ) { // we're looking for a thresh xing
                        if ( v >= thresh ) { // met or exceeded threshold
                            xthresh(chanii) = 1; // set crossed threshold flag for this chan
                            last(chanii) = v; // update last value for this chan, wait til next ti to decide if this is a peak
                        }
                    }
                    else { // xthresh(chanii)==1, in crossed thresh state, now we're look for a peak
                        if ( v > last(chanii) ) // if signal is still increasing
                            last(chanii) = v; // update last value for this chan, wait til next ti to decide if this is a peak
                        else { // signal is decreasing, declare previous ti as a spike timepoint


                            // TODO: I think the next line is wrong!!!!!!!!!!!!!!!!! spikei should start from 0 shouldn't it?


                            spikei = totalnspikes++; // 0-based spike index. assign, then increment
                            // find max chan, ie chan spike is centered on
                            maxchanii = chanii; // start with assumption that current chan is max chan
                            for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all indices into chans again
                                if ( absdata(chanjj, ti) > absdata(maxchanii, ti) )
                                    maxchanii = chanjj; // update maxchanii
                            }
                            maxchani = chans(maxchanii);
                            spiketis(0, spikei) = ti-1; // save previous time index as that of the spikei'th spike
                            spiketis(1, spikei) = maxchani; // store chan spike is centered on
                            // apply spatial and temporal lockouts
                            for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all indices into chans again
                                chanj = chans(chanjj);
                                if ( dm(maxchani, chanj) <= slock ) { // chanjj is within spatial lockout in um
                                    xthresh(chanjj) = 0; // clear its threshx flag
                                    last(chanjj) = 0.0; // reset last so it's ready when it comes out of lockout
                                    // apply its temporal lockout
                                    if ( chanjj > chanii ) // we haven't encountered chanjj yet in the chanii loop
                                        lock(chanjj) = tilock+1; // lockout by one extra timepoint which it'll
                                                                 // then promptly decr before we leave this ti
                                    else
                                        lock(chanjj) = tilock;
                                }
                            }
                            if ( totalnspikes >= maxnspikes ) {
                                return_val = totalnspikes;
                                return return_val; // exit here, don't search any more timepoints
                            }
                            // don't break out of chanii loop to move to next ti: there may be other
                            // chans with spikes that are outside the spatial lockout.
                        }
                    }
                }
            }
        }
        return_val = totalnspikes;
        """
        tCloop = time.clock()
        totalnspikes = weave.inline(code, ['chans', 'nchans', 'absdata', 'nt',
                                           'totalnspikes', 'maxnspikes',
                                           'thresh', 'xthresh', 'last', 'lock',
                                           'tilock', 'slock', 'dm', 'spiketis'],
                                    type_converters=weave.converters.blitz,
                                    compiler='gcc')
        print 'C loop took %.3f sec' % (time.clock()-tCloop)
        nnewspikes = totalnspikes - self.totalnspikes
        self.totalnspikes = totalnspikes # update
        spiketis = spiketis[:, :nnewspikes] # keep only the entries that were filled
        spiketimes = wave.ts[spiketis[0]] # convert from indices to actual spike times, wave.ts is int64 array
        maxchans = spiketis[1]
        return (spiketimes, maxchans)


    def searchchan_weave(self, abschan):
        """Search a single chan of absval data for thresh xings, apply temporal lockout.
        If this crashes, it might be possible that self.spiketis was init'd too small"""
        nt = len(abschan)
        totalnspikes = self.totalnspikes # total num of spikes found by this Detector so far
        maxnspikes = self.maxnspikes
        thresh = self.fixedthresh
        tilock = self.tilock
        spiketis = self.spiketis # init'd in self.search()

        code = r"""
        #line 399 "detect.py" // for debugging
        double last=0.0; // last signal value, uV
        int ti=0; // current time index
        while ( ti<nt && totalnspikes < maxnspikes ) { // enforce maxnspikes across single chan
            if ( abschan(ti) >= thresh ) { // if we've exceeded threshold
                while ( abschan(ti) > last ) { // while signal is still increasing
                    last = abschan(ti); // update last
                    ti++; // go to next timepoint
                }
                // now signal is decreasing, save last timepoint as spike
                spiketis(totalnspikes++ - 1) = ti-1; // 0-based index into spiketis
                last = 0.0; // reset for search for next spike
                ti += tilock+1; // skip forward one temporal lockout, and go to next timepoint
            }
            else
                ti++; // no thresh xing, go to next timepoint
        }
        return_val = totalnspikes;
        """
        totalnspikes = weave.inline(code, ['abschan', 'nt', 'totalnspikes', 'maxnspikes', 'thresh', 'tilock', 'spiketis'],
                                    type_converters=weave.converters.blitz,
                                    compiler='gcc')
        nnewspikes = totalnspikes - self.totalnspikes
        self.totalnspikes = totalnspikes # update
        return spiketis[:nnewspikes]

