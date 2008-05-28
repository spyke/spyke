# include from the python headers
include "Python.pxi"
# include the Numpy C API for use via Cython extension code
include "numpy.pxi"
# initialize numpy - this MUST be done before any other code is executed.
import_array()


cpdef class BipolarAmplitudeFixedThresh_Cy(object): # maybe this shouldn't inherit

    def searchblock(self, wave):
        """Search across all chans in a manageable block of waveform
        data and return a tuple of spike time and maxchan arrays.
        Apply both temporal and spatial lockouts

        TODO: is there an abs f'n in C I can use? what's the most optimized way of taking abs?
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
        thresh = self.thresh

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
              that spike. But iteration isn't really possible, because we have our
              big ti loop to obey and its associated t and s locks.
              Maybe this is just an inherent limit to simple thresh detector?
              Maybe compromise is to search for maxchan immed on thresh xing (within slock),
              then wait til find peak on that chan, then search for maxchan (within slock)
              again upon finding peak.

              - find thresh xing
              - search immed for maxchan within slock
              - apply slock immed around maxchan so you stop getting more thresh xings on other chans for the same spike
              - wait for peak on maxchan
              - once peak found, search again for maxchan within slock
              - reapply slock around maxchan

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

        TODO (maybe): chanii loop should go in random order on each ti, to prevent a chan from
              dominating with its spatial lockout or something like that

        TODO: check if preincrement (++i) is faster than postincrement (i++) in for loops

        """

        # in-place abs f'n in C:
        """
        for ( int ti=0; ti<nt; ti++ ) { // iterate over all timepoint indices
            for ( int chanii=0; chanii<nchans; chanii++ ) { // iterate over all chan indices
                if ( absdata(chanii, ti) < 0 )
                    absdata(chanii, ti) *= -1; // this could be dangerous - unless we do a copy in python, we'll be overwriting actual data. Ah but we do do a copy in Stream.__getitem__ when we concatenate
                    record waveforms, so we'll only be overwriting data from this one slice of Stream, which is fine. The original data will still all be there in the .waveform attrib of the record
            }
        }
        """

        # Finds max chanii at current ti
        # TODO: this should really be applied over and over, recentering the search radius, until maxchanii stops changing
        # requires: chans[], nchans, maxchanii, dm[], slock, absdata[]
        FINDBIGGERCHANCODE = r"""
        // BEGIN FINDBIGGERCHANCODE
        maxchani = chans(maxchanii);
        for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all chan indices
            chanj = chans(chanjj);
            if ( dm(maxchani, chanj) <= slock && // only consider chanjjs within slock of maxchani
                 absdata(chanjj, ti) > absdata(maxchanii, ti) )
                    maxchanii = chanjj; // update maxchanii
                    maxchanchanged = 1;
                    break; // out of this chanjj loop
        maxchanchanged = 0;
        }
        // END FINDBIGGERCHANCODE
        """

        # Applies spatiotemporal lockout centered on current maxchanii from current ti forward
        APPLYLOCKOUTCODE = r"""
        // BEGIN APPLYLOCKOUTCODE
        for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all chan indices
            maxchani = chans(maxchanii);
            chanj = chans(chanjj);
            if ( dm(maxchani, chanj) <= slock ) { // chanjj is within spatial lockout in um
                xthresh(chanjj) = 0; // clear its threshx flag
                last(chanjj) = 0.0; // reset last so it's ready when it comes out of lockout
                // apply its temporal lockout
                if ( chanjj > chanii ) // we haven't encountered chanjj yet in the outer chanii loop
                    lock(chanjj) = tilock+1; // lockout by one extra timepoint which it'll decr before we leave this ti
                else
                    lock(chanjj) = tilock;
            }
        }
        // END APPLYLOCKOUTCODE
        """

        # The main C loop
        CODE = r"""
        #line 385 "detect.py" // for debugging
        int spikei, maxchanii, maxchani, chanj, maxchanchanged;
        double v; // current signal voltage, uV (Python float), using a pointer doesn't seem faster
        for ( int ti=0; ti<nt; ti++ ) { // iterate over all timepoints
            for ( int chanii=0; chanii<nchans; chanii++ ) { // iterate over indices into chans
                if ( lock(chanii) > 0 ) // if this chan is still locked out
                    lock(chanii)--; // decr this chan's temporal lockout
                else { // search for a thresh xing or a peak
                    v = absdata(chanii, ti);
                    if ( xthresh(chanii) == 0 ) { // we're looking for a thresh xing
                        if ( v >= thresh ) { // met or exceeded threshold
                            maxchanii = chanii; // start with assumption that current chan is max chan
                            maxchanchanged = 1;
                            while ( maxchanchanged == 1 ) {
                                %(FINDBIGGERCHAN)s // find maxchanii within slock of chanii
                            }
                            %(APPLYLOCKOUT)s // apply spatiotemporal lockout to prevent extra thresh xings for this developing spike
                            xthresh(maxchanii) = 1; // set crossed threshold flag for this maxchan
                            last(maxchanii) = v; // update last value for this maxchan
                        }
                    }
                    else { // xthresh(chanii)==1, in crossed thresh state, now we're look for a peak
                        if ( v > last(chanii) ) // if signal is still increasing
                            last(chanii) = v; // update last value for this chan, wait til next ti to decide if this is a peak
                        else { // signal is decreasing, declare previous ti as a spike timepoint
                            spikei = totalnspikes++; // 0-based spike index. assign, then increment
                            ti--; // temporarily make last ti the current ti. TODO: hope this is OK messing with the loop counter
                            maxchanii = chanii; // start with assumption that current chan is max chan
                            maxchanchanged = 1;
                            while ( maxchanchanged == 1 ) {
                                %(FINDBIGGERCHAN)s // find maxchan within slock of maxchanni
                            }
                            spiketis(0, spikei) = ti; // save previous time index as that of the spikei'th spike
                            spiketis(1, spikei) = chans(maxchanii); // store chan spike is centered on
                            ti++; // restore current ti
                            %(APPLYLOCKOUT)s // apply spatiotemporal lockout to prevent extra thresh xings for this found spike
                            if ( totalnspikes >= maxnspikes ) {
                                return_val = totalnspikes;
                                return return_val; // exit here, don't search any more timepoints
                            }
                            // don't break out of chanii loop to move to next ti: there may be other
                            // chans at this ti with spikes that are outside the spatial lockout
                        }
                    }
                }
            }
        }
        return_val = totalnspikes;
        """ % {'FINDBIGGERCHAN':FINDBIGGERCHANCODE, 'APPLYLOCKOUT':APPLYLOCKOUTCODE}
        tCloop = time.clock()
        self.CODE = CODE # make it available for inspection
        totalnspikes = weave.inline(CODE, ['chans', 'nchans', 'absdata', 'nt',
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
        maxchans = spiketis[1] # convert from chan indices to actual chan ids
        return (spiketimes, maxchans)
