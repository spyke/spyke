"""Spike sorting classes and frame"""

from __future__ import division

__authors__ = 'Martin Spacek, Reza Lotun'

import os
import sys
import time

import wx

import numpy as np

from spyke.core import WaveForm, Gaussian, intround, MAXLONGLONG
from spyke.gui import wxglade_gui
from spyke.gui.plot import SPIKETW

SPIKETRANGE = (-SPIKETW/2, SPIKETW/2)
MAXCHANTOLERANCE = 100 # um

# save all Spike waveforms, even for those that have never been plotted or added to a template
SAVEALLSPIKEWAVES = False

SPLITTERSASH = 360
SORTSPLITTERSASH = 117
SPIKESORTPANELWIDTHPERCOLUMN = 120
SORTFRAMEHEIGHT = 950


class Sort(object):
    """A spike sorting session, in which you can do multiple Detection runs,
    build Templates up from spikes in those Detection runs, and then use Templates
    to sort Spikes.
    Formerly known as a Session, and before that, a Collection.
    A .sort file is a single Sort object, pickled and gzipped"""
    def __init__(self, detector=None, probe=None, stream=None):
        self.detector = detector # this sort session's current Detector object
        self.probe = probe # only one probe design per sort allowed
        self.detections = {} # history of detection runs
        self.stream = stream
        # all unsorted spikes detected in this sort session across all Detection runs, indexed by unique ID
        # sorted spikes go in their respective template's .spikes dict
        self.spikes = {}
        self.templates = {} # first hierarchy of templates
        self.trash = {} # discarded spikes

        self._detid = 0 # used to count off unqiue Detection run IDs
        self._spikeid = 0 # used to count off unique Spike IDs
        self._templid = 0 # used to count off unique Template IDs

    def get_stream(self):
        return self._stream

    def set_stream(self, stream=None):
        """Set Stream object for self's detector and all detections,
        for (un)pickling purposes"""
        try:
            stream.sampfreq = self.sampfreq # restore sampfreq and shcorrect to stream
            stream.shcorrect = self.shcorrect
        except AttributeError:
            pass # either stream is None or self.sampfreq/shcorrect aren't bound
        self._stream = stream
        self.detector.stream = stream
        for detection in self.detections.values():
            detection.detector.stream = stream

    stream = property(get_stream, set_stream)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        if self.stream != None:
            d['sampfreq'] = self.stream.sampfreq # grab sampfreq and shcorrect before removing stream
            d['shcorrect'] = self.stream.shcorrect
        del d['_stream'] # don't pickle the stream, cuz it relies on ctsrecords, which rely on open .srf file
        return d

    def append_spikes(self, spikes):
        """Append spikes to self.
        Don't add a new spike from a new detection if the identical spike
        (same maxchan and t) is already in self.spikes"""
        newspikes = set(spikes.values()).difference(self.spikes.values())
        duplicates = set(spikes.values()).difference(newspikes)
        if duplicates:
            print 'not adding duplicate spikes %r' % [ spike.id for spike in duplicates ]
        uniquespikes = {}
        for newspike in newspikes:
            uniquespikes[newspike.id] = newspike
        self.spikes.update(uniquespikes)
        return uniquespikes

    def match(self, templates=None, weighting='signal', sort=True):
        """Match templates to all .spikes with nearby maxchans,
        save error values to respective templates.

        Note: slowest step by far is loading in the wave data from disk.
        (First match is slow, subsequent ones are ~ 15X faster.)
        Unless something's done about that in advance, don't bother optimizing here much.
        Right now, once waves are loaded, performance is roughly 20000 matches/sec

        TODO: Nick's alternative to gaussian distance weighting: have two templates: a mean template, and an stdev
        template, and weight the error between each matched spike and the mean on each chan at each timepoint by
        the corresponding stdev value (divide the error by the stdev, so that timepoints with low stdev are more
        sensitive to error)

        TODO: looks like I still need to make things more nonlinear - errors at high signal values aren't penalized enough,
        while errors at small signal values are penalized too much. Try cubing both signals, then taking sum(err**2)

        DONE: maybe even better, instead of doing an elaborate cubing of signal, followed by a rather elaborate
        gaussian spatiotemporal weighting of errors, just take difference of signals, and weight the error according
        to the abs(template_signal) at each point in time and across chans. That way, error in parts of the signal far from
        zero are considered more important than deviance of perhaps similar absolute value for signal close to zero

        """
        templates = templates or self.templates.values() # None defaults to matching all templates
        sys.stdout.write('matching')
        t0 = time.clock()
        nspikes = len(self.spikes)
        dm = self.detector.dm
        for template in templates:
            template.err = [] # overwrite any existing .err attrib
            trange = template.trange
            templatewave = template.wave[template.chans] # pull out template's enabled chans
            #stdev = template.get_stdev()[template.chans] # pull out template's enabled chans
            #stdev[stdev == 0] = 1 # replace any 0s with 1s - TODO: what's the best way to avoid these singularities?
            weights = template.get_weights(weighting=weighting, sstdev=self.detector.slock/2,
                                           tstdev=self.detector.tlock/2) # Gaussian weighting in space and/or time
            for spike in self.spikes.values():
                # check if spike.maxchan is outside some minimum distance from template.maxchan
                if dm[template.maxchan, spike.maxchan] > MAXCHANTOLERANCE: # um
                    continue # don't even bother
                if spike.wave.data == None or template.trange != SPIKETRANGE: # make sure their data line up
                    spike.update_wave(trange) # this slows things down a lot, but is necessary
                # slice template's enabled chans out of spike, calculate sum of squared weighted error
                # first impression is that dividing by stdev makes separation worse, not better
                #err = (templatewave.data - spike.wave[template.chans].data) / stdev * weights # low stdev means more sensitive to error
                spikewave = spike.wave[template.chans] # pull out template's enabled chans from spike
                if weighting == 'signal':
                    weights = np.abs(np.asarray([templatewave.data, spikewave.data])).max(axis=0) # take elementwise max of abs of template and spike data
                err = (templatewave.data - spikewave.data) * weights # weighted error
                err = (err**2).sum(axis=None) # sum of squared weighted error
                template.err.append((spike.id, intround(err)))
            template.err = np.asarray(template.err, dtype=np.int64)
            if sort and len(template.err) != 0:
                i = template.err[:, 1].argsort() # row indices that sort by error
                template.err = template.err[i]
            sys.stdout.write('.')
        print '\nmatch took %.3f sec' % (time.clock()-t0)


class Detection(object):
    """A spike detection run, which happens every time Search is pressed.
    When you're merely searching for the previous/next spike with
    F2/F3, that's not considered a detection run"""
    def __init__(self, sort, detector, id=None, datetime=None, sms=None):
        self.sort = sort # parent sort session
        self.detector = detector # Detector object used in this Detection run
        self.id = id
        self.datetime = datetime
        self.sms = sms # list of valid SpikeModels collected from Detector.search

    def set_spikeids(self):
        """Give each SpikeModel an ID, inc sort's _spikeid counter after each one.
        Stick a references to all SpikeModels into a .spikes dict, using spike IDs as the keys"""
        self.spikes = {}
        for sm in self.sms:
            sm.id = self.sort._spikeid
            self.sort._spikeid += 1 # inc for next unique SpikeModel
            self.spikes[sm.id] = sm
        '''
        for t, chan in self.spikes_array.T: # same as iterate over cols of non-transposed spikes array
            s = Spike(self.sort._spikeid, chan, t, self)
            self.sort._spikeid += 1 # inc for next unique Spike
            self.spikes[s.id] = s
        '''
    '''
    deprecated: use sm.chans instead:
    def get_slock_chans(self, maxchan):
        """Get or generate list of chans within spatial lockout of maxchan, use
        spatial lockout of self.detector
        Note this can't be used as the getter in a property, I think cuz you can't pass
        args to a getter"""
        try:
            return self._slock_chans[maxchan]
        except KeyError:
            det = self.detector
            # NOTE: dm is now always a full matrix, where its row indices always correspond
            # to channel indices, so no need for messing around with indices into indices...
            #chans = np.asarray(det.chans) # chans that correspond to rows/columns in det.dm
            #maxchani, = np.where(chans == maxchan) # get index into det.dm that corresponds to maxchan
            chans, = np.where(det.dm[maxchan].flat <= det.slock) # flat removes the singleton dimension
            chans = list(chans)
            self._slock_chans[maxchan] = chans # save for quick retrieval next time
            return chans
    '''
    def __eq__(self, other):
        """Compare detection runs by their .sms lists
        TODO: see if there's any overlap between self.sms and other.sms, ie duplicate spikes,
        and raise a warning in a dialog box or something
        """
        return np.all(self.sms == other.sms)


class Template(object):
    """A collection of spikes that have been deemed somehow, whether manually
    or automatically, to have come from the same cell. A Template's waveform
    is the mean of its member spikes"""
    def __init__(self, sort, id=None, parent=None):
        self.sort = sort # parent sort session
        self.id = id # template id
        self.parent = parent # parent template, if self is a subtemplate
        self.subtemplates = None
        self.wave = WaveForm() # init to empty waveform
        self.maxchan = None
        self.chans = None # chans enabled for plotting/matching/ripping
        self.spikes = {} # member spikes that make up this template
        self.trange = SPIKETRANGE
        self.t = 0 # relative reference timestamp, a bit redundant, here for symmetry with Spike.t
        self.plot = None # Plot currently holding self
        self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        #self.surffname # not here, let's allow templates to have spikes from different files?

    def update_wave(self):
        """Update mean waveform, should call this every time .spikes or .trange
        are modified.
        Setting .trange as a property to do so automatically works.
        Setting .spikes as a property to do so automatically doesn't work, because
        properties only catch name binding of spikes, not modification of an object
        that's already been bound"""
        if self.spikes == {}: # no member spikes
            self.wave = WaveForm() # empty waveform
            return
        spike = self.spikes.values()[0] # get a random member spike
        if spike.wave.data == None: # make sure its WaveForm isn't empty
            spike.update_wave(trange=self.trange)
        #self.maxchan = self.maxchan or spike.maxchan # set maxchan if it hasn't been already
        self.chans = self.chans or spike.chans # set enabled chans if they haven't been already
        ts = self.wave.ts # see if they've already been set
        if ts == None:
            ts = spike.wave.ts - spike.t # timestamps relative to self.t=0
        lo, hi = ts.searchsorted(self.trange)
        ts = ts[lo:hi] # slice them according to trange

        wavechans = self.wave.chans or spike.wave.chans # chan ids that correspond to rows in wave.data
        data = []
        for spike in self.spikes.values():
            # check each spike for timepoints that don't match up, update the spike so that they do
            # note: spike is no longer just some random member spike as it was above
            if spike.wave.ts == None or ((spike.wave.ts - spike.t) != ts).all():
                spike.update_wave(trange=self.trange)
            assert spike.wave.chans == wavechans # being really thorough here...
            data.append(spike.wave.data) # collect spike's data
        data = np.asarray(data).mean(axis=0)
        self.wave.data = data
        self.wave.ts = ts
        #print 'template[%d].wave.ts = %r' % (self.id, ts)
        self.wave.chans = wavechans # could be None, to indicate to WaveForm that data is contiguous and complete
        self.maxchan = self.get_maxchan()
        #self.chans = spike.detection.get_slock_chans(self.maxchan) # from random spike's detection
        return self.wave

    def get_stdev(self):
        """Return 2D array of stddev of each timepoint of each chan of member spikes.
        Assumes self.update_wave has already been called"""
        data = []
        for spike in self.spikes.values():
            data.append(spike.wave.data) # collect spike's data
        stdev = np.asarray(data).std(axis=0)
        return stdev

    def get_chans(self):
        return self._chans

    def set_chans(self, chans):
        """Update maxchan on replacement of enabled chans.
        User chan selection should trigger search for maxchan"""
        self._chans = chans
        self.maxchan = self.get_maxchan()

    chans = property(get_chans, set_chans)

    def get_maxchan(self):
        """Find maxchan at t=0 in mean waveform, constrained to enabled chans

        Notes:
            - don't recenter self.chans on maxchan, leave actual chan selection to user
            - maxchan should however be constrained to currently enabled chans
        """
        if self.wave.data == None or self.chans == None:
            return None
        data = self.wave.data
        ts = self.wave.ts
        t0i, = np.where(ts == 0) # find column index that corresponds to t=0
        assert len(t0i) == 1 # make sure there's only one reference timepoint
        maxchani = abs(data[self.chans, t0i]).argmax() # find index into self.chans with greatest abs(signal) at t=0
        #maxchani = abs(data).max(axis=1).argmax() # ignore sign, find max across columns, find row with greatest max
        maxchan = self.chans[maxchani] # dereference
        return maxchan

    def get_trange(self):
        return self._trange

    def set_trange(self, trange):
        """Reset self's time range relative to t=0 spike time,
        update slice of member spikes, and update mean waveform"""
        self._trange = trange
        for spike in self.spikes.values():
            spike.update_wave(trange=trange)
        self.update_wave()

    trange = property(get_trange, set_trange)

    def get_weights(self, weighting=None, sstdev=None, tstdev=None):
        """Returns unity, spatial, temporal, or spatiotemporal Gaussian weights
        for self's enabled chans in self.wave.data, given spatial and temporal
        stdevs"""
        nchans = len(self.chans)
        nt = len(self.wave.data[self.chans[0]]) # assume all chans have the same number of timepoints
        if weighting == None:
            weights = 1
        elif weighting == 'spatial':
            weights = self.get_gaussian_spatial_weights(sstdev) # vector
        elif weighting == 'temporal':
            weights = self.get_gaussian_temporal_weights(tstdev) # vector
        elif weighting == 'spatiotemporal':
            sweights = self.get_gaussian_spatial_weights(sstdev)
            tweights = self.get_gaussian_temporal_weights(tstdev)
            weights = np.outer(sweights, tweights) # matrix, outer product of the two
        elif weighting == 'signal':
            weights = None # this is handled by caller
        #print '\nweights:\n%r' % weights
        return weights

    def get_gaussian_spatial_weights(self, stdev):
        """Return a vector that weights self.chans according to a 2D gaussian
        centered on self.maxchan with standard deviation stdev in um"""
        g = Gaussian(mean=0, stdev=stdev)
        d = self.sort.detector.dm[self.maxchan, self.chans] # distances between maxchan and all enabled chans
        weights = g[d]
        weights.shape = (-1, 1) # vertical vector with nchans rows, 1 column
        return weights

    def get_gaussian_temporal_weights(self, stdev):
        """Return a vector that weights timepoints in self's mean waveform
        by a gaussian centered on t=0, with standard deviation stdev in us"""
        g = Gaussian(mean=0, stdev=stdev)
        ts = self.wave.ts # template mean timepoints relative to t=0 spike time
        weights = g[ts] # horizontal vector with 1 row, nt timepoints
        return weights

    '''
    def __del__(self):
        """Is this run on 'del template'?"""
        for spike in self.spikes:
            spike.template = None # remove self from all spike.template fields

    def pop(self, spikeid):
        return self.spikes.pop(spikeid)
    '''
    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        d['plot'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        d['itemID'] = None # clear tree item ID, since that'll have changed anyway on unpickle
        return d

'''
class Spike(object):
    """Either an unsorted spike, or a member spike in a Template,
    or a sorted spike in a Detection (or should that be sort session?)"""
    def __init__(self, id, maxchan, t, detection):
        self.id = id # some integer for easy user identification
        self.maxchan = maxchan
        self.t = t # absolute timestamp, generally falls within span of waveform
        self.detection = detection # Detection run self was detected on
        self.chans = self.detection.get_slock_chans(maxchan) # chans enabled for plotting/matching/matchripping
        self.template = None # template object it belongs to, None means self is an unsorted spike
        self.wave = WaveForm() # init to empty waveform
        self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        self.plot = None # Plot currently holding self
        # try loading it right away on init, instead of waiting until plot, see if it's fast enough
        # nah, too slow after doing an OnSearch, don't load til plot or til Save (ie pickling)
        #self.update_wave()

    def update_wave(self, trange=SPIKETRANGE):
        """Load/update self's waveform, defaults to default spike time window centered on self.t"""
        self.wave = self[self.t+trange[0] : self.t+trange[1]]
        return self.wave

    def __eq__(self, other):
        """Spikes are considered identical if they have the
        same timepoint and the same maxchan"""
        return self.t == other.t and self.maxchan == other.maxchan

    def __hash__(self):
        """Unique hash value for self, based on .t and .maxchan.
        Required for effectively using spikes in a Set"""
        return hash((self.t, self.maxchan)) # hash of their tuple, should guarantee uniqueness

    def __getitem__(self, key):
        """Return WaveForm for this spike given slice key"""
        assert key.__class__ == slice
        stream = self.detection.detector.stream
        if stream != None: # stream is available
            self.wave = stream[key] # let stream handle the slicing, save result
            return self.wave
        elif self.wave != None: # stream unavailable, .wave from before last pickling is available
            return self.wave[key] # slice existing .wave
        else: # neither stream nor existing .wave available
            return WaveForm() # return empty waveform

    def __getstate__(self):
        """Get object state for pickling"""
        if SAVEALLSPIKEWAVES and self.wave.data == None:
            # make sure .wave is loaded before pickling to file
            self.update_wave()
        d = self.__dict__.copy()
        d['plot'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        d['itemID'] = None # clear tree item ID, since that'll have changed anyway on unpickle
        return d
'''

class Match(object):
    """Holds all the settings of a match run. A match run is when you compare each
    template to all of the detected but unsorted spikes in Sort.spikes, plot an
    error histogram for each template, and set the error threshold for each to
    decide which spikes match the template. Fast, simple, no noise spikes to worry
    about, but is susceptible to spike misalignment. Compare with a Rip"""

    def match(self):
        pass


class Rip(object):
    """Holds all the Rip settings. A rip is when you take each template and
    slide it across the entire file. A spike is detected and
    sorted at timepoints where the error between template and file falls below
    some threshold. Slow, and requires distinguishing a whole lotta noise spikes"""

    def rip(self):
        pass


class MatchRip(Match, Rip):
    """A hybrid of the two. Rip each template across all of the unsorted spikes
    instead of across the entire file. Compared to a Match, a MatchRip can better
    handle unsorted unspikes that are misaligned, with the downside that you now
    have a lot of noise spikes to distinguish as well, but not as many as in a normal Rip"""

    def matchrip(self):
        pass


class SortFrame(wxglade_gui.SortFrame):
    """Sort frame"""
    def __init__(self, *args, **kwargs):
        wxglade_gui.SortFrame.__init__(self, *args, **kwargs)
        self.spykeframe = self.Parent
        ncols = self.sort.probe.ncols
        size = (SPLITTERSASH + SPIKESORTPANELWIDTHPERCOLUMN * ncols,
                SORTFRAMEHEIGHT)
        self.SetSize(size)
        self.splitter.SetSashPosition(SPLITTERSASH) # do this here because wxGlade keeps messing this up
        self.sort_splitter.SetSashPosition(SORTSPLITTERSASH)

        self.listTimer = wx.Timer(owner=self.list)

        self.lastSelectedListSpikes = []
        self.lastSelectedTreeObjects = []

        columnlabels = ['sID', 'chan', 'time', 'err'] # spike list column labels
        for coli, label in enumerate(columnlabels):
            self.list.InsertColumn(coli, label)
        for coli in range(len(columnlabels)): # this needs to be in a separate loop it seems
            self.list.SetColumnWidth(coli, wx.LIST_AUTOSIZE_USEHEADER) # resize columns to fit

        self.list.Bind(wx.EVT_TIMER, self.OnListTimer)
        self.list.Bind(wx.EVT_RIGHT_DOWN, self.OnListRightDown)
        self.list.Bind(wx.EVT_KEY_DOWN, self.OnListKeyDown)
        #self.tree.Bind(wx.EVT_LEFT_DOWN, self.OnTreeLeftDown) # doesn't fire when clicking on non focused item, bug #4448
        self.tree.Bind(wx.EVT_LEFT_UP, self.OnTreeLeftUp) # need this to catch clicking on non focused item, bug #4448
        self.tree.Bind(wx.EVT_RIGHT_DOWN, self.OnTreeRightDown)
        self.tree.Bind(wx.EVT_KEY_DOWN, self.OnTreeKeyDown)
        self.tree.Bind(wx.EVT_KEY_UP, self.OnTreeKeyUp)
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnTreeSelectChanged)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def get_sort(self):
        return self.spykeframe.sort

    def set_sort(self):
        raise RuntimeError, "SortFrame's .sort not settable"

    sort = property(get_sort, set_sort) # make this a property for proper behaviour after unpickling

    def OnSize(self, evt):
        """Re-save reflines_background after resizing the frame"""
        # resize doesn't actually happen until after this handler exits,
        # so we have to CallAfter
        wx.CallAfter(self.DrawRefs)
        evt.Skip()

    def OnSplitterSashChanged(self, evt):
        """Re-save reflines_background after resizing the SortPanel(s)
        with the frame's primary splitter"""
        print 'in OnSplitterSashChanged'
        wx.CallAfter(self.DrawRefs)

    def OnClose(self, evt):
        frametype = self.__class__.__name__.lower().replace('frame', '') # remove 'Frame' from class name
        self.spykeframe.HideFrame(frametype)

    def OnListSelect(self, evt):
        """Restart list selection timer
        listTimer explanation: on any selection event, start or restart the timer for say 1 msec.
        Then, when timer runs down, run self.list.GetSelections() and compare to previous list of
        selections, and execute your plots accordingly. This makes all the sel event handling fast"""
        self.listTimer.Stop()
        self.listTimer.Start(milliseconds=1, oneShot=True) # only fire one timer event after specified interval

    def OnListDeselect(self, evt):
        self.OnListSelect(evt)

    def OnListTimer(self, evt):
        """Run when started timer runs out and triggers a TimerEvent"""
        selectedRows = self.list.GetSelections()
        selectedListSpikes = [ self.listRow2Spike(row) for row in selectedRows ]
        removeSpikes = [ spike for spike in self.lastSelectedListSpikes if spike not in selectedListSpikes ]
        addSpikes = [ spike for spike in selectedListSpikes if spike not in self.lastSelectedListSpikes ]
        self.RemoveObjectsFromPlot(removeSpikes)
        self.AddObjects2Plot(addSpikes)
        self.lastSelectedListSpikes = selectedListSpikes # save for next time

    def OnListRightDown(self, evt):
        """Toggle selection of the clicked list item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection ListEvent happening before the MouseEvent, or something"""
        print 'in OnListRightDown'
        pt = evt.GetPosition()
        itemID, flags = self.list.HitTest(pt)
        spike = self.listRow2Spike(itemID)
        print 'spikeID is %r' % spike.id
        # this would be nice, but doesn't work (?) cuz apparently somehow the
        # selection ListEvent happens before MouseEvent that caused it:
        #selected = not self.list.IsSelected(itemID)
        #self.list.Select(itemID, on=int(not selected))
        # here is a yucky workaround:
        try:
            self.spikesortpanel.used_plots['s'+str(spike.id)] # is it plotted?
            selected = True # if so, item must be selected
            print 'spike %d in used_plots' % spike.id
        except KeyError:
            selected = False # item is not selected
            print 'spike %d not in used_plots' % spike.id
        self.list.Select(itemID, on=not selected) # toggle selection, this fires sel spike, which updates the plot

    def OnListColClick(self, evt):
        coli = evt.GetColumn()
        if coli == 0:
            self.SortListByID()
        elif coli == 1:
            self.SortListByChan()
        elif coli == 2:
            self.SortListByTime()
        elif coli == 3:
            self.SortListByErr()
        else:
            raise ValueError, 'weird column id %d' % coli

    def OnListKeyDown(self, evt):
        """Spike list key down evt"""
        key = evt.GetKeyCode()
        if key == wx.WXK_TAB:
            self.tree.SetFocus() # change focus to tree
        elif key in [wx.WXK_SPACE, wx.WXK_RETURN]:
            self.list.ToggleFocusedItem()
            return # evt.Skip() seems to prevent toggling, or maybe it untoggles
        elif key in [ord('A'), wx.WXK_LEFT]:
            self.MoveCurrentSpikes2Template(which='selected')
        elif key in [ord('C'), ord('N'), ord('T')]: # wx.WXK_SPACE doesn't seem to work
            self.MoveCurrentSpikes2Template(which='new')
        elif key in [wx.WXK_DELETE, ord('D')]:
            self.MoveCurrentSpikes2Trash()
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        evt.Skip()

    def OnTreeSelectChanged(self, evt=None):
        """Due to bugs #2307 and #626, a SEL_CHANGED event isn't fired when
        (de)selecting the currently focused item in a tree with the wx.TR_MULTIPLE
        flag set, as it is here. So, this handler has to be called manually on mouse
        and keyboard events. Don't think there's any harm in calling this handler twice
        for all other cases where a SEL_CHANGED event is fired"""
        print 'in OnTreeSelectChanged'
        if evt: # just use this as a hack to update currently focused
            item = evt.GetItem()
            if item:
                self.tree._focusedItem = item
                print 'currently focused item: %s' % self.tree.GetItemText(item)
            return # don't allow the selection event to actually happen?????????????
        self.tree._selectedItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        print [ self.tree.GetItemText(item) for item in self.tree._selectedItems ]
        selectedTreeObjects = [] # objects could be a mix of Spikes and Templates
        for itemID in self.tree._selectedItems:
            item = self.tree.GetItemPyData(itemID)
            selectedTreeObjects.append(item)
        removeObjects = [ obj for obj in self.lastSelectedTreeObjects if obj not in selectedTreeObjects ]
        addObjects = [ obj for obj in selectedTreeObjects if obj not in self.lastSelectedTreeObjects ]
        self.RemoveObjectsFromPlot(removeObjects)
        self.AddObjects2Plot(addObjects)
        self.lastSelectedTreeObjects = selectedTreeObjects # save for next time

    def OnTreeLeftDown(self, evt):
        print 'in OnTreeLeftDown'
        pt = evt.GetPosition()
        itemID, flags = self.tree.HitTest(pt)
        if itemID.IsOk(): # if we've clicked on an item
            # leave selection event uncaught, call selection handler
            # after OS has finished doing the actual (de)selecting
            wx.CallAfter(self.OnTreeSelectChanged)

    def OnTreeLeftUp(self, evt):
        """Need this to catch clicking on non focused item, bug #4448"""
        print 'in OnTreeLeftUp'
        self.OnTreeLeftDown(evt)

    def OnTreeRightDown(self, evt):
        """Toggle selection of the clicked item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection TreeEvent happening before the MouseEvent"""
        print 'in OnTreeRightDown'
        pt = evt.GetPosition()
        itemID, flags = self.tree.HitTest(pt)
        if not itemID.IsOk(): # if we haven't clicked on an item
            return
        obj = self.tree.GetItemPyData(itemID) # either a Spike or a Template
        # first, restore all prior selections in the tree (except our item) that were cleared by the right click selection event
        for itemID in self.tree._selectedItems: # rely on tree._selectedItems being judiciously kept up to date
            self.tree.SelectItem(itemID)
        if obj.itemID not in self.tree._selectedItems: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(obj.itemID, select=False)
        self.OnTreeSelectChanged() # now plot accordingly

    def OnTreeSpaceUp(self, evt):
        """Toggle selection of the clicked item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection TreeEvent happening before the SPACE KeyUp event.
        This causes an annoying flicker of the currently selected items, because they
        all unfortunately get deselected on the uncatchable SPACE keydown event,
        and aren't reselected until the SPACE keyup event"""
        print 'in OnTreeSpaceUp'
        itemID = self.tree.GetFocusedItem()
        obj = self.tree.GetItemPyData(itemID) # either a Spike or a Template
        # first, restore all prior selections in the tree (except our item) that were cleared by the space selection event
        for itemID in self.tree._selectedItems: # rely on tree._selectedItems being judiciously kept up to date
            self.tree.SelectItem(itemID)
        if obj.itemID not in self.tree._selectedItems: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(obj.itemID, select=False)
        self.OnTreeSelectChanged() # now plot accordingly

    def OnTreeKeyDown(self, evt):
        key = evt.GetKeyCode()
        #print 'key down: %r' % key
        if key == wx.WXK_TAB:
            self.list.SetFocus() # change focus to list
        elif key == wx.WXK_RETURN: # space only triggered on key up, see bug #4448
            self.tree.ToggleFocusedItem()
            #wx.CallAfter(self.OnTreeSelectChanged)
            self.OnTreeSelectChanged()
            # evt.Skip() seems to prevent toggling, or maybe it untoggles
        elif key in [wx.WXK_DELETE, ord('D'),]:
            self.MoveCurrentObjects2List()
        elif key == ord('A'): # allow us to add from spike list even if tree is in focus
            self.MoveCurrentSpikes2Template(which='selected')
        elif key in [ord('C'), ord('N'), ord('T')]: # ditto for creating a new template
            self.MoveCurrentSpikes2Template(which='new')
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        elif key in [wx.WXK_UP, wx.WXK_DOWN]: # keyboard selection hack around multiselect bug
            wx.CallAfter(self.OnTreeSelectChanged)
            #self.OnTreeSelectChanged()
        self.tree._selectedItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnTreeKeyUp(self, evt):
        key = evt.GetKeyCode()
        #print 'key up: %r' % key
        if key == wx.WXK_SPACE: # space only triggered on key up, see bug #4448
            if evt.ControlDown():
                #wx.CallAfter(self.OnTreeSelectChanged)
                self.OnTreeSelectChanged()
            else:
                #self.tree.ToggleFocusedItem()
                #wx.CallAfter(self.OnTreeSelectChanged)
                #self.OnTreeSelectChanged()
                self.OnTreeSpaceUp(evt)
                #evt.Skip() seems to prevent toggling, or maybe it untoggles
        self.tree._selectedItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnSortTree(self, evt):
        root = self.tree.GetRootItem()
        if root: # tree isn't empty
            self.tree.SortChildren(root)
            self.RelabelTemplates(root)

    def OnMatchTemplate(self, evt):
        """Match spikes in spike list against first selected template, populate err column"""
        errcol = 3 # err is in 3rd column (0-based)
        template = self.GetFirstSelectedTemplate()
        if not template: # no templates selected
            return
        self.sort.match(templates=[template])
        sid2err = dict(template.err) # maps spike ID to its error for this template
        for rowi in range(self.list.GetItemCount()):
            sid = int(self.list.GetItemText(rowi))
            try:
                err = str(sid2err[sid])
            except KeyError: # no err for this sid because the spike and template don't overlap enough
                err = ''
            erritem = self.list.GetItem(rowi, errcol)
            erritem.SetText(err)
            self.list.SetItem(erritem)

    def RelabelTemplates(self, root):
        """Consecutively relabel templates according to their vertical order in the TreeCtrl.
        Relabeling happens both in the TreeCtrl and in the in .sort.templates dict"""
        templates = self.tree.GetTreeChildrenPyData(root) # get all children in order from top to bottom
        self.sort.templates = {} # clear the dict, gc won't kick in cuz we still have a ref
        for templatei, template in enumerate(templates):
            template.id = templatei # update its id
            self.sort.templates[template.id] = template # add it to its key in template dict
            self.tree.SetItemText(template.itemID, 't'+str(template.id)) # update its entry in the tree
        self.sort._templid = templatei + 1 # reset unique Template ID counter to make next added template consecutive

    def SortListByID(self):
        """Sort spike list by spike ID"""
        for rowi in range(self.list.GetItemCount()):
            sid = int(self.list.GetItemText(rowi))
            self.list.SetItemData(rowi, sid)
        self.list.SortItems(cmp) # now do the actual sort, based on the item data

    def SortListByChan(self):
        """Sort spike list by ycoord of spike maxchans, from top to bottom of probe"""
        # first set the itemdata for each row
        SiteLoc = self.sort.probe.SiteLoc
        for rowi in range(self.list.GetItemCount()):
            sid = int(self.list.GetItemText(rowi)) # 0th column
            s = self.sort.spikes[sid]
            xcoord = SiteLoc[s.maxchan][0]
            ycoord = SiteLoc[s.maxchan][1]
            # hack to make items sort by ycoord, or xcoord if ycoords are identical
            data = intround((ycoord + xcoord/1000)*1000) # needs to be an int unfortunately
            self.list.SetItemData(rowi, data)
        self.list.SortItems(cmp) # now do the actual sort, based on the item data

    def SortListByTime(self):
        """Sort spike list by spike timepoint"""
        for rowi in range(self.list.GetItemCount()):
            sid = int(self.list.GetItemText(rowi)) # 0th column
            t = self.sort.spikes[sid].t
            # this will cause a problem once timestamps exceed 2**32 us, see SortListByErr for fix
            self.list.SetItemData(rowi, t)
        self.list.SortItems(cmp) # now do the actual sort, based on the item data

    def SortListByErr(self):
        """Sort spike list by match error.
        Hack to get around stupid SetItemData being limited to int32s"""
        errcol = 3 # err is in 3rd column (0-based)
        errs = []
        for rowi in range(self.list.GetItemCount()):
            err = self.list.GetItem(rowi, errcol).GetText()
            try:
                err = int(err)
            except ValueError: # err is empty string
                err = MAXLONGLONG
            errs.append(err)
        erris = np.asarray(errs).argsort() # indices that return errs sorted
        for rowi, erri in enumerate(erris):
            self.list.SetItemData(erri, rowi) # the erri'th row is set the rowi'th rank value
        self.list.SortItems(cmp) # now do the actual sort, based on the item data

    def DrawRefs(self):
        """Redraws refs and resaves background of sort panel(s)"""
        self.spikesortpanel.draw_refs()
        #self.chartsortpanel.draw_refs()

    def Append2SpikeList(self, spikes):
        """Append spikes to self's spike list control"""
        SiteLoc = self.sort.probe.SiteLoc
        for s in spikes.values():
            row = [str(s.id), s.maxchan, s.t] # leave err column empty for now
            self.list.Append(row)
            # using this instead of .Append(row) is just as slow:
            #rowi = self.list.InsertStringItem(sys.maxint, str(s.id))
            #self.list.SetStringItem(rowi, 1, str(s.maxchan))
            #self.list.SetStringItem(rowi, 2, str(s.t))
            # should probably use a virtual listctrl to speed up listctrl creation
            # and subsequent addition and especially removal of items
            xcoord = SiteLoc[s.maxchan][0]
            ycoord = SiteLoc[s.maxchan][1]
            # hack to make items sort by ycoord, or xcoord if ycoords are identical
            data = intround((ycoord + xcoord/1000)*1000) # needs to be an int unfortunately
            # use item count instead of counting from 0 cuz you want to handle there
            # already being items in the list from prior append/removal
            self.list.SetItemData(self.list.GetItemCount()-1, data)
        self.list.SortItems(cmp) # sort the list by maxchan from top to bottom of probe
        #width = wx.LIST_AUTOSIZE_USEHEADER # resize columns to fit
        # hard code column widths for precise control, autosize seems buggy
        for coli, width in {0:40, 1:40, 2:80, 3:60}.items(): # (sid, chan, time, err)
            self.list.SetColumnWidth(coli, width)

    def AddObjects2Plot(self, objects):
        #print 'objects to add: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.addObjects(objects)
        #self.chartsortpanel.addObjects(objects)

    def RemoveObjectsFromPlot(self, objects):
        #print 'objects to remove: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.removeObjects(objects)
        #self.chartsortpanel.removeObjects(objects)

    def UpdateObjectsInPlot(self, objects):
        #print 'objects to update: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.updateObjects(objects)
        #self.chartsortpanel.updateObjects(objects)

    #TODO: should self.OnTreeSelectChanged() (update plot) be called more often at the end of many of the following methods?:

    def CreateTemplate(self):
        """Create, select, and return a new template"""
        template = Template(self.sort, self.sort._templid, parent=None)
        self.sort._templid += 1 # inc for next unique Template
        self.sort.templates[template.id] = template # add template to sort session
        self.AddTemplate2Tree(template)
        return template

    def AddTemplate2Tree(self, template):
        """Add a template to the tree control"""
        root = self.tree.GetRootItem()
        if not root.IsOk(): # if tree doesn't have a valid root item
            root = self.tree.AddRoot('Templates')
        template.itemID = self.tree.AppendItem(root, 't'+str(template.id)) # add template to tree
        self.tree.SetItemPyData(template.itemID, template) # associate template tree item with template

    def DeleteTemplate(self, template):
        """Move a template's spikes back to the spike list, delete it
        from the tree, and remove it from the sort session"""
        for spike in template.spikes.values():
            self.MoveSpike2List(spike)
        self.tree.Delete(template.itemID)
        del self.sort.templates[template.id]

    def listRow2Spike(self, row):
        """Return Spike at list row"""
        spikei = int(self.list.GetItemText(row))
        spike = self.sort.spikes[spikei]
        return spike

    def MoveSpike2Template(self, spike, row, template=None):
        """Move a spike spike from unsorted sort.spikes to a template.
        Also, move it from a list control row to a template in the tree.
        If template is None, create a new one
        """
        # make sure this spike isn't already a member of this template,
        # or of any other template
        for templ in self.sort.templates.values():
            if spike in templ.spikes.values():
                print "Can't move: spike %d is identical to a member spike in template %d" % (spike.id, templ.id)
                return
        self.list.DeleteItem(row) # remove it from the spike list
        self.list.Select(row) # automatically select the new item at that position
        createdTemplate = False
        if template == None:
            template = self.CreateTemplate()
            createdTemplate = True
        del self.sort.spikes[spike.id] # remove spike from unsorted sort.spikes
        template.spikes[spike.id] = spike # add spike to template
        template.update_wave() # update mean template waveform
        spike.template = template # bind template to spike
        self.AddSpike2Tree(template.itemID, spike)
        if createdTemplate:
            #self.tree.Expand(root) # make sure root is expanded
            self.tree.Expand(template.itemID) # expand template
            self.tree.UnselectAll() # unselect all items in tree
            self.tree.SelectItem(template.itemID) # select the newly created template
            self.OnTreeSelectChanged() # now plot accordingly
        return template

    def MoveSpike2Trash(self, spike, row):
        """Move spike from spike list to trash"""
        self.list.DeleteItem(row) # remove it from the spike list
        self.list.Select(row) # automatically select the new item at that position
        del self.sort.spikes[spike.id] # remove spike from unsorted sort.spikes
        self.sort.trash[spike.id] = spike # add it to trash
        print 'moved spike %d to trash' % spike.id

    def AddSpike2Tree(self, parent, spike):
        """Add a spike to the tree, where parent is a tree itemID"""
        spike.itemID = self.tree.AppendItem(parent, 's'+str(spike.id)) # add spike to tree, save its itemID
        self.tree.SetItemPyData(spike.itemID, spike) # associate spike tree item with spike

    def MoveSpike2List(self, spike):
        """Move a spike spike from a template in the tree back to the list control"""
        # make sure this spike isn't already in sort.spikes
        if spike in self.sort.spikes.values():
            # would be useful to print out the guilty spike id in the spike list, but that would require a more expensive search
            print "Can't move: spike %d (maxchan=%d, t=%d) in template %d is identical to an unsorted spike in the spike list" \
                  % (spike.id, spike.maxchan, spike.t, spike.template.id)
            return
        self.tree.Delete(spike.itemID)
        template = spike.template
        del template.spikes[spike.id] # del spike from its template's spike dict
        self.sort.spikes[spike.id] = spike # restore spike to unsorted sort.spikes
        template.update_wave() # update mean template waveform
        spike.template = None # unbind spike's template from spike
        spike.itemID = None # no longer applicable
        data = [spike.id, spike.maxchan, spike.t]
        self.list.InsertRow(0, data) # stick it at the top of the list, is there a better place to put it?
        # TODO: maybe re-sort the list

    def MoveCurrentSpikes2Template(self, which='selected'):
        if which == 'selected':
            template = self.GetFirstSelectedTemplate()
        elif which == 'new':
            template = None # indicates we want a new template
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        for row in selected_rows:
            spike = self.listRow2Spike(row)
            if spike.wave.data != None: # only move it to template if it's got wave data
                template = self.MoveSpike2Template(spike, row, template) # if template was None, it isn't any more
            else:
                print "can't add spike %d to template because its data isn't accessible" % spike.id
        if template != None and template.plot != None: # if it exists and it's plotted
            self.UpdateObjectsInPlot([template]) # update its plot

    def MoveCurrentObjects2List(self):
        for itemID in self.tree.GetSelections():
            if itemID: # check if spike's tree parent (template) has already been deleted
                obj = self.tree.GetItemPyData(itemID)
                if obj.__class__ == Spike:
                    self.MoveSpike2List(obj)
                elif obj.__class__ == Template:
                    self.DeleteTemplate(obj)
        self.OnTreeSelectChanged() # update plot

    def MoveCurrentSpikes2Trash(self):
        """Move currently selected spikes in spike list to trash"""
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        for row in selected_rows:
            spike = self.listRow2Spike(row)
            self.MoveSpike2Trash(spike, row)

    def GetFirstSelectedTemplate(self):
        for itemID in self.tree.GetSelections():
            obj = self.tree.GetItemPyData(itemID)
            if obj.__class__ == Template:
                return obj
            # no template selected, check to see if an spike is selected in the tree, grab its template
            elif obj.__class__ == Spike:
                return obj.template
        return None
