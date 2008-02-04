from __future__ import division

""" Provides convenient stream interface to .srf files """

__authors__ = ['Martin Spacek', 'Reza Lotun']

import spyke.surf
import numpy as np

DEFAULTINTERPSAMPFREQ = 50000 # default interpolated sample rate, in Hz


class Stream(object):
    """ Streaming object. Maps from timestamps to record index of stream
    data to retrieve the approriate range of waveform data from disk.
    """
    def __init__(self, records=None, sampfreq=None):
        """ Takes a sorted temporal (not necessarily evenly-spaced, due to
        pauses in recording) sequence of ContinuousRecords: either
        HighPassRecords or LowPassMultiChanRecords
        """
        self.records = records

        # array of record timestamps
        self.rts = np.asarray([record.TimeStamp for record in self.records])

        # if no sampfreq passed in, use sampfreq of the raw data
        self.sampfreq = sampfreq or records[0].layout.sampfreqperchan

    def __len__(self):
        """ Total number of timepoints? Length in time? Interp'd or raw? """
        raise NotImplementedError()

    def __getitem__(self, key, endinclusive=False):
        """ Called when Stream object is indexed into using [] or with a slice
        object, indicating start and end timepoints in us. Returns the
        corresponding WaveForm object, which has as its attribs the 2D
        multichannel waveform array as well as the timepoints,
        potentially spanning multiple ContinuousRecords
        """

        # for now, accept only slice objects as keys
        assert key.__class__ == slice

        # find the first and last records corresponding to the slice. If the
        # start of the slice matches a record's timestamp, start with that
        # record. If the end of the slice matches a record's timestamp, end
        # with that record (even though you'll only potentially use the one
        # timepoint from that record, depending on the value of 'endinclusive')
        lorec, hirec = self.rts.searchsorted([key.start, key.stop],
                                                                side='right')

        # we always want to get back at least 1 record (ie records[0:1]). When
        # slicing, we need to do lower bounds checking (don't go less than 0),
        # but not upper bounds checking
        cutrecords = self.records[max(lorec-1, 0):max(hirec, 1)]
        for record in cutrecords:
            try:
                record.waveform
            except AttributeError:
                # to save time, only load the waveform if not already loaded
                record.load()

        # join all waveforms, returns a copy
        data = np.concatenate([record.waveform for record in cutrecords],
                                axis=1)
        try:

            # all highpass records should be using the same layout,
            # use tres from the first one
            tres = cutrecords[0].layout.tres

        except AttributeError:
            # records are lowpassmulti
            tres = cutrecords[0].tres

        # build up waveform timepoints, taking into account any time gaps in
        # between records due to pauses in recording
        ts = []
        for record in cutrecords:
            tstart = record.TimeStamp
            # number of timepoints (columns) in this record's waveform
            nt = record.waveform.shape[-1]
            ts.extend(range(tstart, tstart + nt*tres, tres))

            #del record.waveform
            # save memory by unloading waveform data from records that
            # aren't needed anymore
        ts = np.asarray(ts)
        lo, hi = ts.searchsorted([key.start, key.stop])
        data = data[:, lo:hi+endinclusive]
        ts = ts[lo:hi+endinclusive]

        # interp and s+h correct here
        data, ts = self.interp(data, ts, self.sampfreq)

        # transform AD values to uV
        extgain = self.records[0].layout.extgain
        intgain = self.records[0].layout.intgain
        data = self.ADVal_to_uV(data, intgain, extgain)

        # return a WaveForm object
        return WaveForm(data=data, ts=ts, sampfreq=self.sampfreq)


    def ADVal_to_uV(self, adval, intgain, extgain):
        """ Convert AD values to micro-volts """
        #Round((ADValue - 2048)*(10 / (2048
        #                 * ProbeArray[m_ProbeIndex].IntGain
        #                 * ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan]))
        #                 * V2uV);
        return (adval - 2048) * (10 / (2048 * intgain * extgain[0]) * 1e6)

    def interp(self, data, ts, sampfreq=None, kind='nyquist'):
        """ Returns interpolated and sample-and-hold corrected data and
        timepoints, at the given sample frequency
        """
        if kind == 'nyquist':
            # do Nyquist interpolation and S+H correction here
            # find a scipy function that'll do Nyquist interpolation?
            # TODO: Implement this!
            return data, ts
        else:
            raise ValueError, 'Unknown kind of interpolation %r' % kind

    def plot(self, chanis=None, trange=None):
        """ Creates a simple matplotlib plot of the specified chanis over
        trange
        """
        # wouldn't otherwise need this
        import pylab as pl
        from pylab import get_current_fig_manager as gcfm

        try:
            # see if neuropy is available
            from neuropy.Core import lastcmd, neuropyScalarFormatter, \
                                                        neuropyAutoLocator
        except ImportError:
            pass

        if chanis == None:
            # all high pass records should have the same chanlist
            if self.records[0].__class__ == HighPassRecord:
                chanis = self.records[0].layout.chanlist
                # same goes for lowpassmultichanrecords, each has its own
                # set of chanis, derived previously from multiple
                # single layout.chanlists
            elif self.records[0].__class__ == LowPassMultiChanRecord:
                chanis = self.records[0].chanis
            else:
                raise ValueError, 'unknown record type %s in self.records' % \
                                                    self.records[0].__class__
        nchans = len(chanis)

        if trange == None:
            trange = (self.rts[0], self.rts[0]+100000)

        # make a waveform object
        wf = self[trange[0]:trange[1]]
        figheight = 1.25+0.2*nchans
        self.f = pl.figure(figsize=(16, figheight))
        self.a = self.f.add_subplot(111)

        try:
            gcfm().frame.SetTitle(lastcmd())
        except NameError:
            pass
        except AttributeError:
            pass

        try:

            # better behaved tick label formatter
            self.formatter = neuropyScalarFormatter()

            # use a thousands separator
            self.formatter.thousandsSep = ','

            # better behaved tick locator
            self.a.xaxis.set_major_locator(neuropyAutoLocator())
            self.a.xaxis.set_major_formatter(self.formatter)

        except NameError:
            pass
        for chanii, chani in enumerate(chanis):
            # upcast to int32 to prevent int16 overflow
            self.a.plot(wf.ts/1e3,
                    (np.int32(wf.data[chanii])-2048+500*chani)/500., '-',
                                                            label=str(chani))
        #self.a.legend()
        self.a.set_xlabel('time (ms)')
        self.a.set_ylabel('channel id')

        # assumes chanis are sorted
        self.a.set_ylim(chanis[0]-1, chanis[-1]+1)
        bottominches = 0.75
        heightinches = 0.15+0.2*nchans
        bottom = bottominches / figheight
        height = heightinches / figheight
        self.a.set_position([0.035, bottom, 0.94, height])
        pl.show()


class WaveForm(object):
    """ Waveform object, has data, timestamps and sample frequency attribs """
    def __init__(self, data=None, ts=None, sampfreq=None):
        self.data = data # potentially multichannel, depending on shape
        self.ts = ts # timestamps array, one for each sample (column) in data
        self.sampfreq = sampfreq

