        #line 321 "detect.py" // for debugging
        double last=0.0; // last signal value, uV
        int nnewspikes=0; // num new spikes found in this f'n
        int ti=0; // current time index
        while ( ti<nt && totalnspikes < maxnspikes ) { // enforce maxnspikes across single chan
            if ( abschan(ti) >= thresh ) { // if we've exceeded threshold
                while ( abschan(ti) > last ) { // while signal is still increasing
                    last = abschan(ti); // update last
                    ti++; // go to next timepoint
                }
                // now signal is decreasing, save last timepoint as spike
                totalnspikes++;
                nnewspikes++;
                spiketis(nnewspikes-1) = ti-1; // 0-based index into spiketis
                last = 0.0; // reset for search for next spike
                ti += tilock+1; // skip forward one temporal lockout, and go to next timepoint
            }
            else
                ti++; // no thresh xing, go to next timepoint
        }
        return_val = nnewspikes;


        #line 254 "detect.py" // for debugging
        int nnewspikes=0; // num new spikes found in this f'n
        int ti=0; // current time index
        int chan=0; // current chan index
        int spikei=0; // current spike index for current chan
        double v; // current signal voltage, uV
        for ( ti=0; ti<nt && totalnspikes<maxnspikes; ti++ ) {
            // chan loop should go in random order, not numerical, to prevent a chan from
            // dominating with its spatial lockout. Also, there might be missing channels
            for ( chan=0; chan<nchans; chan++ ) {
                if ( lock(chan) > 0 ) // if this chan is still locked out
                    lock(chan)--; // decr this chan's temporal lockout
                else { // search for a thresh xing or a peak
                    v = absdata(chan, ti); // TODO: v should be a pointer to prevent this copy operation?
                    if ( xthresh(chan) == 0 ) { // we're looking for a thresh xing
                        if ( v >= thresh && xthresh(chan) == 0 ) { // met or exceeded threshold and
                                                                   // xthresh flag hasn't been set yet
                                                                   // TODO: check if >= is slower than >
                            xthresh(chan) = 1; // set crossed threshold flag for this chan
                            last(chan) = v; // update last value for this chan, have to wait til
                                            // next ti to decide if this is a peak
                        }
                    }
                    else { // we've found a thresh xing, now we're look for a peak
                        if ( v > last(chan) ) // if signal is still increasing
                            last(chan) = v; // update last value for this chan, have to wait til
                                            // next ti to decide if this is a peak
                        else { // signal is decreasing, save previous ti as spike
                            spikei = nspikes(chan)++; // 0-based spike index. assign, then increment
                            totalnspikes++;
                            nnewspikes++;
                            spiketis(chan, spikei) = ti-1; // save previous time index as that of the nth spike for this chan
                            xthresh(chan) = 0; // we've found the peak, clear crossed thresh flag for this chan
                            last(chan) = 0.0; // reset for search for next spike
                            lock(chan) = tilock; // apply temporal lockout for this chan
                            //lock(chan and nearby chans) = tilock; // set temporal lockout for this chan and all chans
                                                                  // within slock distance of it
                        }
                    }
                }
            }
        }
        return_val = nnewspikes;
