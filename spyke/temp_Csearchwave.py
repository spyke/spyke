import numpy as np
from scipy import weave
from scipy.weave import converters

def Csearchwave(abschan, thresh, tilockout):
    """Maybe limit this to a single chan, deal with spatial lockout in python,
    do peak searching and temporal lockout on a single chan basis in C. Or, maybe use
    C for both chan and ti loop, but leave spatial lockout to a later step (which itself
    could be done in another C weave f'n)

    abschan is an array of floats

    """

    thresh = float(thresh)
    assert tilockout.__class__ == int
    assert tilockout >= 0 # num of timepoint indices to lock out after a spike

    nt = len(abschan)
    spikeis = np.zeros(nt/2, dtype=np.int32) # holds spike times, cant have more than one max every other timepoint

    code = r"""
    #line 19 "detect.py" // (this is only useful for debugging)
    double last=0.0; // last signal value, uV
    int nspikes=0;
    int ti=0; // current time index
    while ( ti<nt ) {
        if (abschan(ti) >= thresh) {
            while (abschan(ti) > last) { // signal is still increasing
                last = abschan(ti);
                ti++; // go to next timepoint
            }
            // signal is decreasing, save last timepoint as spike
            nspikes++;
            spikeis(nspikes-1) = ti-1;
            last = 0.0; // reset for search for next spike
            ti += tilockout; // skip forward one temporal lockout
        }
        else
            ti++; // no thresh xing, go to next timepoint
    }
    return_val = nspikes;
    """
    nspikes = weave.inline(code, ['abschan', 'nt', 'thresh', 'tilockout', 'spikeis'],
                           type_converters=converters.blitz,
                           compiler='gcc')
    spikeis = spikeis[:nspikes]
    return spikeis


abschan = np.array([ 2. ,  1. ,  0. ,  1. ,  2. ,  1. ,  0. ,  0. ,  0. ,  0. ,  5. ,  0. ,  0. ,  6.2,  2.3,  0. ,  0. ,  2. ])
Csearchwave(abschan, thresh=1, tilockout=1)
