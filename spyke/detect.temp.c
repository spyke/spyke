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
                            spikei = totalnspikes++; // 0-based spike index. assign, then increment
                            // find max chan, ie chan spike is centered on
                            maxchanii = chanii; // start with assumption that current chan is max chan
                            for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all indices into chans again
                                if ( absdata(chanjj, ti) > absdata(maxchanii, ti) )
                                    maxchanii = chanjj; // update maxchanii
                            }
                            spiketis(0, spikei) = ti-1; // save previous time index as that of the spikei'th spike
                            spiketis(1, spikei) = maxchanii; // store chan spike is centered on
                            // apply spatial and temporal lockouts
                            for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all indices into chans again
                                maxchani = chans(maxchanii);
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
