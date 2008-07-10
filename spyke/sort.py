"""Spike sorting classes and frame"""

from __future__ import division

__authors__ = 'Martin Spacek, Reza Lotun'

import os
import wx

import numpy as np

from spyke.core import WaveForm, intround
from spyke.gui import wxglade_gui
from spyke.gui.plot import DEFEVENTTW

# save all Event waveforms, even for those that have never been plotted or added to a template
SAVEALLEVENTWAVES = False

SPLITTERSASH = 300
SORTSPLITTERSASH = 117
SPIKESORTPANELWIDTHPERCOLUMN = 120
SORTFRAMEHEIGHT = 950


class Session(object):
    """A spike sorting session, in which you can do multiple Detection runs,
    build Templates up from events in those Detection runs, and then use Templates
    to sort spike Events.
    Formerly known as a Collection.
    A .sort file is a single sort Session object, pickled and gzipped"""
    def __init__(self, detector=None, probe=None, stream=None):
        self.detector = detector # this session's current Detector object
        self.probe = probe # only one probe design per session allowed
        self.detections = [] # history of detection runs, in chrono order, reuse deleted Detection IDs
        self.stream = stream
        # all unsorted events detected in this sort session across all Detection runs, indexed by unique ID
        # sorted events go in their respective template's .events dict
        self.events = {}
        self.templates = {} # first hierarchy of templates
        self.trash = {} # discarded events

        self._detid = 0 # used to count off unqiue Detection run IDs
        self._eventid = 0 # used to count off unique Event IDs
        self._templid = 0 # used to count off unique Template IDs

    def get_stream(self):
        return self._stream

    def set_stream(self, stream=None):
        """Set Stream object for self's detector and all detections,
        for unpickling purposes"""
        # Enforce a single fixed .tres and .sampfreq per Session object
        # This means that the first stream that's set cannot be None
        try:
            self.tres
            self.sampfreq
        except AttributeError:
            self.tres = stream.tres
            self.sampfreq = stream.sampfreq
        self._stream = stream
        self.detector.stream = stream
        for detection in self.detections:
            detection.detector.stream = stream

    stream = property(get_stream, set_stream)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['_stream'] # don't pickle the stream, cuz it relies on ctsrecords, which rely on open .srf file
        return d

    def append_events(self, events):
        """Append events to self.
        Don't add a new event from a new detection if the identical event
        (same maxchan and t) is already in session.events"""
        newevents = set(events.values()).difference(self.events.values())
        uniqueevents = {}
        for newevent in newevents:
            uniqueevents[newevent.id] = newevent
        self.events.update(uniqueevents)
        return uniqueevents


class Detection(object):
    """A spike detection run, which happens every time Search is pressed.
    When you're merely searching for the previous/next spike with
    F2/F3, that's not considered a detection run"""
    def __init__(self, session, detector, id=None, datetime=None, events_array=None):
        self.session = session # parent sort Session
        self.detector = detector # Detector object used in this Detection run
        self.id = id
        self.datetime = datetime
        self.events_array = events_array # 2D array output of Detector.search
        self._slock_chans = {}
        #self.spikes = {} # a dict of Event objects? a place to note which events in this detection have been chosen as either member spikes of a template or sorted spikes. Need this here so we know which Spike objects to delete from this sort Session when we delete a Detection

    def set_events(self):
        """Convert .events_array to dict of Event objects, inc session's _eventid counter"""
        self.events = {}
        for t, chan in self.events_array.T: # same as iterate over cols of non-transposed events array
            e = Event(self.session._eventid, chan, t, self)
            self.session._eventid += 1 # inc for next unique Event
            self.events[e.id] = e

    def get_slock_chans(self, maxchan):
        """Get or generate list of chans within spatial lockout of maxchan, use
        spatial lockout of self.detector
        Note this can't be used as the getter in a property, I think cuz you can't pass
        args in a getter"""
        try:
            return self._slock_chans[maxchan]
        except KeyError:
            det = self.detector
            chans = np.asarray(det.chans) # chans that correspond to rows/columns in det.dm
            maxchani, = np.where(chans == maxchan) # get index into det.dm that corresponds to maxchan
            chanis, = np.where(det.dm[maxchani].flat <= det.slock) # flat removes the singleton dimension
            chans = list(chans[chanis]) # chans within slock of maxchan
            self._slock_chans[maxchan] = chans # save for quick retrieval next time
            return chans

    def __eq__(self, other):
        """Compare detection runs by their .events_array"""
        # TODO: see if there's any overlap between self.events and other.events, ie duplicate events,
        # and raise a warning in a dialog box or something
        return np.all(self.events_array == other.events_array)


class Template(object):
    """A collection of spikes that have been deemed somehow, whether manually
    or automatically, to have come from the same cell. A Template's waveform
    is the mean of its member spikes"""
    def __init__(self, session, id=None, parent=None):
        self.session = session # parent sort Session
        self.id = id # template id
        self.parent = parent # parent template, if self is a subtemplate
        self.subtemplates = None
        self.maxchan = None
        self.chans = None # chans enabled for plotting/ripping
        self.events = {} # member spike events that make up this template
        self.trange = (-DEFEVENTTW/2, DEFEVENTTW/2)
        self.t = 0 # relative reference timestamp, a bit redundant, here for symmetry with Event.t
        self.wave = WaveForm() # init to empty waveform
        self.plot = None # Plot currently holding self
        self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        #self.surffname # not here, let's allow templates to have spikes from different files?

    def update_wave(self):
        """Update mean waveform, should call this every time .events is modified.
        Setting .events as a property to do so automatically doesn't work, because
        properties only catch name binding events, not modification of an object
        that's already been bound"""
        if self.events == {}: # no member spikes
            self.wave = WaveForm() # empty waveform
            return
        else: # has member spikes
            self.maxchan = self.maxchan or self.events.values()[0].maxchan # set maxchan if it hasn't been already
            self.chans = self.chans or self.events.values()[0].chans # set enabled chans if they haven't been already
        data = []
        relts = np.arange(self.trange[0], self.trange[1], self.session.tres) # timestamps relative to spike time
        event = self.events.values()[0] # grab a random event
        if event.wave.data == None: # make sure its waveform isn't empty
            event.update_wave(trange=self.trange)
        sampfreq = self.wave.sampfreq or event.wave.sampfreq
        chan2i = self.wave.chan2i or event.wave.chan2i
        for event in self.events.values():
            # check each event for timepoints that don't match up, update the event so that they do
            if event.wave.ts == None or not ((event.wave.ts - event.t) == relts).all():
                event.update_wave(trange=self.trange)
            assert event.wave.sampfreq == sampfreq # being really thorough here...
            assert event.wave.chan2i == chan2i
            data.append(event.wave.data)
        data = np.asarray(data).mean(axis=0)
        # TODO: search data and find maxchan, set self.maxchan. Or not, just leave it up to user to select chans
        self.wave.data = data
        self.wave.ts = relts
        self.wave.sampfreq = sampfreq
        self.wave.chan2i = chan2i
        return self.wave

    def get_trange(self):
        return self._trange

    def set_trange(self, trange=(-DEFEVENTTW/2, DEFEVENTTW/2)):
        """Reset self's time range relative to t=0 spike time,
        update slice of member spikes, and update mean waveform"""
        self._trange = trange
        for event in self.events.values():
            event.update_wave(trange=trange)
        self.update_wave()

    trange = property(get_trange, set_trange)

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


class Event(object):
    """Either an unsorted event, or a member spike in a Template,
    or a sorted spike in a Detection (or should that be sort Session?)"""
    def __init__(self, id, maxchan, t, detection):
        self.id = id # some integer for easy user identification
        self.maxchan = maxchan
        self.t = t # absolute timestamp, generally falls within span of waveform
        self.detection = detection # Detection run self was detected on
        self.chans = self.detection.get_slock_chans(maxchan)
        self.template = None # template object it belongs to, None means self is an unsorted event
        self.wave = WaveForm() # init to empty waveform
        self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        self.plot = None # Plot currently holding self
        #self.cluster = None # cluster run this Spike was sorted on
        #self.rip = None # rip this Spike was sorted on
        # try loading it right away on init, instead of waiting until plot, see if it's fast enough
        # nah, too slow after doing an OnSearch, don't load til plot or til Save (ie pickling)
        #self.update_wave()

    def update_wave(self, trange=(-DEFEVENTTW/2, DEFEVENTTW/2)):
        """Load/update self's waveform, defaults to default event time window centered on self.t"""
        self.wave = self[self.t+trange[0] : self.t+trange[1]]
        return self.wave

    def __eq__(self, other):
        """Events are considered identical if they have the
        same timepoint and the same maxchan"""
        return self.t == other.t and self.maxchan == other.maxchan

    def __hash__(self):
        """Unique hash value for self, based on .t and .maxchan.
        Required for effectively using Events in a Set"""
        return hash((self.t, self.maxchan)) # hash of their tuple, should guarantee uniqueness

    def __getitem__(self, key):
        """Return WaveForm for this event given slice key"""
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
        if SAVEALLEVENTWAVES and self.wave.data == None:
            # make sure .wave is loaded before pickling to file
            self.update_wave()
        d = self.__dict__.copy()
        d['plot'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        d['itemID'] = None # clear tree item ID, since that'll have changed anyway on unpickle
        return d


class Match(object):
    """Holds all the settings of a match run. A match run is when you compare each
    template to all of the detected but unsorted spikes in Session.events, plot an
    error histogram for each template, and set the error threshold for each to
    decide which events match the template. Fast, simple, no noise events to worry
    about, but is susceptible to spike misalignment. Compare with a Rip"""

    def match(self):
        pass


class Rip(object):
    """Holds all the Rip settings. A rip is when you take each template and
    slide it across the entire file. A spike is detected and
    sorted at timepoints where the error between template and file falls below
    some threshold. Slow, and requires distinguishing a whole lotta noise events"""

    def rip(self):
        pass


class MatchRip(Match, Rip):
    """A hybrid of the two. Rip each template across all of the unsorted spikes
    instead of across the entire file. Compared to a Match, a MatchRip can better
    handle unsorted unspikes that are misaligned, with the downside that you now
    have a lot of noise events to distinguish as well, but not as many as in a normal Rip"""

    def matchrip(self):
        pass

# or just have two options in the Sort pane: Rip against: detected events; entire file


class SortFrame(wxglade_gui.SortFrame):
    """Sort frame"""
    def __init__(self, *args, **kwargs):
        wxglade_gui.SortFrame.__init__(self, *args, **kwargs)
        self.spykeframe = self.Parent
        ncols = self.session.probe.ncols
        size = (SPLITTERSASH + SPIKESORTPANELWIDTHPERCOLUMN * ncols,
                SORTFRAMEHEIGHT)
        self.SetSize(size)

        self.listTimer = wx.Timer(owner=self.list)

        self.lastSelectedListEvents = []
        self.lastSelectedTreeObjects = []

        columnlabels = ['eID', 'chan', 'time'] # event list column labels
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
        #self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnTreeSelectChanged)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def get_session(self):
        return self.spykeframe.session

    def set_session(self):
        raise RuntimeError, "SortFrame's .session not settable"

    session = property(get_session, set_session) # make this a property for proper behaviour after unpickling

    def OnSize(self, evt):
        """Put code here to re-save reflines_background"""
        # resize doesn't actually happen until after this handler exits,
        # so we have to CallAfter
        wx.CallAfter(self.DrawRefs)
        evt.Skip()

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
        selectedListEvents = [ self.listRow2Event(row) for row in selectedRows ]
        removeEvents = [ event for event in self.lastSelectedListEvents if event not in selectedListEvents ]
        addEvents = [ event for event in selectedListEvents if event not in self.lastSelectedListEvents ]
        self.RemoveObjectsFromPlot(removeEvents)
        self.AddObjects2Plot(addEvents)
        self.lastSelectedListEvents = selectedListEvents # save for next time

    def OnListRightDown(self, evt):
        """Toggle selection of the clicked list item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection ListEvent happening before the MouseEvent, or something"""
        print 'in OnListRightDown'
        pt = evt.GetPosition()
        itemID, flags = self.list.HitTest(pt)
        event = self.listRow2Event(itemID)
        print 'eventID is %r' % event.id
        # this would be nice, but doesn't work (?) cuz apparently somehow the
        # selection ListEvent happens before MouseEvent that caused it:
        #selected = not self.list.IsSelected(itemID)
        #self.list.Select(itemID, on=int(not selected))
        # here is a yucky workaround:
        try:
            self.spikesortpanel.used_plots['e'+str(event.id)] # is it plotted?
            selected = True # if so, item must be selected
            print 'event %d in used_plots' % event.id
        except KeyError:
            selected = False # item is not selected
            print 'event %d not in used_plots' % event.id
        self.list.Select(itemID, on=not selected) # toggle selection, this fires sel event, which updates the plot

    def OnListColClick(self, evt):
        coli = evt.GetColumn()
        if coli == 0:
            self.SortListByID()
        elif coli == 1:
            self.SortListByChan()
        elif coli == 2:
            self.SortListByTime()
        else:
            raise ValueError, 'weird column id %d' % coli

    def OnListKeyDown(self, evt):
        """Event list key down evt"""
        key = evt.GetKeyCode()
        if key in [ord('A'), wx.WXK_LEFT]: # wx.WXK_RETURN doesn't seem to work
            self.MoveCurrentEvents2Template(which='selected')
        elif key in [ord('C'), ord('N'), ord('T')]: # wx.WXK_SPACE doesn't seem to work
            self.MoveCurrentEvents2Template(which='new')
        elif key in [wx.WXK_DELETE, ord('D')]:
            self.MoveCurrentEvents2Trash()
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        evt.Skip()

    def OnTreeSelectChanged(self, evt=None):
        """Due to bugs #2307 and #626, a SEL_CHANGED event isn't fired when
        deselecting the currently focused item in a tree with the wx.TR_MULTIPLE
        flag set, as it is here. So, this handler has to be called manually on mouse
        and keyboard events"""
        print 'in OnTreeSelectChanged'
        self._selectedTreeItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        selectedTreeObjects = [] # objects could be a mix of Events and Templates
        for itemID in self._selectedTreeItems:
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
        obj = self.tree.GetItemPyData(itemID) # either an Event or a Template
        # first, restore all prior selections in the tree (except our item) that were cleared by the right click selection event
        for itemID in self._selectedTreeItems: # rely on _selectedTreeItems being judiciously kept up to date
            self.tree.SelectItem(itemID)
        if obj.itemID not in self._selectedTreeItems: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(obj.itemID, select=False)
        self.OnTreeSelectChanged() # now plot accordingly

    def OnTreeKeyDown(self, evt):
        key = evt.GetKeyCode()
        #print 'key down: %r' % key
        if key in [wx.WXK_DELETE, ord('D')]:
            self.MoveCurrentObjects2List()
        elif key == ord('A'): # allow us to add from event list even if tree is in focus
            self.MoveCurrentEvents2Template(which='selected')
        elif key in [ord('C'), ord('N'), ord('T')]: # ditto for creating a new template
            self.MoveCurrentEvents2Template(which='new')
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        elif key in [wx.WXK_UP, wx.WXK_DOWN]: # keyboard selection hack around multiselect bug
            wx.CallAfter(self.OnTreeSelectChanged)
        self._selectedTreeItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnTreeKeyUp(self, evt):
        key = evt.GetKeyCode()
        #print 'key up: %r' % key
        if key == wx.WXK_SPACE: # space only triggered on key up, see bug #4448
            wx.CallAfter(self.OnTreeSelectChanged)
        self._selectedTreeItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnSortTree(self, evt):
        root = self.tree.GetRootItem()
        if root: # tree isn't empty
            self.tree.SortChildren(root)
            self.RelabelTemplates(root)

    def RelabelTemplates(self, root):
        templates = self.tree.GetTreeChildrenPyData(root) # get all children in order from top to bottom
        for templatei, template in enumerate(templates):
            del self.session.templates[template.id] # remove template from its old key in template dict
            template.id = templatei # update its id
            self.session.templates[template.id] = template # add it to its (potentially) new key in template dict
            self.tree.SetItemText(template.itemID, 't'+str(template.id)) # update its entry in the tree

    def SortListByID(self):
        """Sort event list by event ID"""
        for rowi in range(self.list.GetItemCount()):
            eid = int(self.list.GetItemText(rowi))
            self.list.SetItemData(rowi, eid)
        # now do the actual sort, based on the item data
        self.list.SortItems(cmp)

    def SortListByChan(self):
        """Sort event list by ycoord of event maxchans,
        from top to bottom of probe"""
        # first set the itemdata for each row
        SiteLoc = self.session.probe.SiteLoc
        for rowi in range(self.list.GetItemCount()):
            eid = int(self.list.GetItemText(rowi))
            e = self.session.events[eid]
            xcoord = SiteLoc[e.maxchan][0]
            ycoord = SiteLoc[e.maxchan][1]
            # hack to make items sort by ycoord, or xcoord if ycoords are identical
            data = intround((ycoord + xcoord/1000)*1000) # needs to be an int unfortunately
            self.list.SetItemData(rowi, data)
        # now do the actual sort, based on the item data
        self.list.SortItems(cmp)

    def SortListByTime(self):
        """Sort event list by event timepoint"""
        for rowi in range(self.list.GetItemCount()):
            eid = int(self.list.GetItemText(rowi))
            t = self.session.events[eid].t
            self.list.SetItemData(rowi, t)
        # now do the actual sort, based on the item data
        self.list.SortItems(cmp)

    def DrawRefs(self):
        """Redraws refs and resaves background of sort panel(s)"""
        self.spikesortpanel.draw_refs()
        #self.chartsortpanel.draw_refs()

    def Append2EventList(self, events):
        """Append events to self's event list control"""
        SiteLoc = self.session.probe.SiteLoc
        for e in events.values():
            row = [str(e.id), e.maxchan, e.t]
            self.list.Append(row)
            # using this instead of .Append(row) is just as slow:
            #rowi = self.list.InsertStringItem(sys.maxint, str(e.id))
            #self.list.SetStringItem(rowi, 1, str(e.maxchan))
            #self.list.SetStringItem(rowi, 2, str(e.t))
            # should probably use a virtual listctrl to speed up listctrl creation
            # and subsequent addition and especially removal of items
            xcoord = SiteLoc[e.maxchan][0]
            ycoord = SiteLoc[e.maxchan][1]
            # hack to make items sort by ycoord, or xcoord if ycoords are identical
            data = intround((ycoord + xcoord/1000)*1000) # needs to be an int unfortunately
            # use item count instead of counting from 0 cuz you want to handle there
            # already being items in the list from prior append/removal
            self.list.SetItemData(self.list.GetItemCount()-1, data)
        self.list.SortItems(cmp) # sort the list by maxchan from top to bottom of probe
        #width = wx.LIST_AUTOSIZE_USEHEADER # resize columns to fit
        # hard code column widths for precise control, autosize seems buggy
        for coli, width in {0:40, 1:40, 2:80}.items(): # (eID, chan, time)
            self.list.SetColumnWidth(coli, width)

    def AddObjects2Plot(self, objects):
        print 'objects to add: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.addObjects(objects)
        #self.chartsortpanel.addObjects(objects)

    def RemoveObjectsFromPlot(self, objects):
        print 'objects to remove: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.removeObjects(objects)
        #self.chartsortpanel.removeObjects(objects)

    def UpdateObjectsInPlot(self, objects):
        print 'objects to update: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.updateObjects(objects)
        #self.chartsortpanel.updateObjects(objects)

    #TODO: should self.OnTreeSelectChanged() (update plot) be called more often at the end of many of the following methods?:

    def CreateTemplate(self):
        """Create, select, and return a new template"""
        template = Template(self.session, self.session._templid, parent=None)
        self.session._templid += 1 # inc for next unique Template
        self.session.templates[template.id] = template # add template to session
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
        """Move a template's events back to the event list, delete it
        from the tree, and remove it from the session"""
        for event in template.events.values():
            self.MoveEvent2List(event)
        self.tree.Delete(template.itemID)
        del self.session.templates[template.id]

    def listRow2Event(self, row):
        """Return Event at list row"""
        eventi = int(self.list.GetItemText(row))
        event = self.session.events[eventi]
        return event

    def MoveEvent2Template(self, event, row, template=None):
        """Move a spike event from unsorted session.events to a template.
        Also, move it from a list control row to a template in the tree.
        If template is None, create a new one
        """
        # make sure this event isn't already a member event of this template,
        # or of any other template
        for templ in self.session.templates.values():
            if event in templ.events.values():
                print "Can't move: event %d is identical to a member event in template %d" % (event.id, templ.id)
                return
        self.list.DeleteItem(row) # remove it from the event list
        self.list.Select(row) # automatically select the new item at that position
        createdTemplate = False
        if template == None:
            template = self.CreateTemplate()
            createdTemplate = True
        del self.session.events[event.id] # remove event from unsorted session.events
        template.events[event.id] = event # add event to template
        template.update_wave() # update mean template waveform
        event.template = template # bind template to event
        self.AddEvent2Tree(template.itemID, event)
        if createdTemplate:
            #self.tree.Expand(root) # make sure root is expanded
            self.tree.Expand(template.itemID) # expand template
            self.tree.UnselectAll() # unselect all items in tree
            self.tree.SelectItem(template.itemID) # select the newly created template
            self.OnTreeSelectChanged() # now plot accordingly
        return template

    def MoveEvent2Trash(self, event, row):
        """Move event from event list to trash"""
        self.list.DeleteItem(row) # remove it from the event list
        self.list.Select(row) # automatically select the new item at that position
        del self.session.events[event.id] # remove event from unsorted session.events
        self.session.trash[event.id] = event # add it to trash
        print 'moved event %d to trash' % event.id

    def AddEvent2Tree(self, parent, event):
        """Add an event to the tree, where parent is a tree itemID"""
        event.itemID = self.tree.AppendItem(parent, 'e'+str(event.id)) # add event to tree, save its itemID
        self.tree.SetItemPyData(event.itemID, event) # associate event tree item with event

    def MoveEvent2List(self, event):
        """Move a spike event from a template in the tree back to the list control"""
        # make sure this event isn't already in session.events
        if event in self.session.events.values():
            # would be useful to print out the guilty event id in the event list, but that would require a more expensive search
            print "Can't move: event %d (maxchan=%d, t=%d) in template %d is identical to an unsorted event in the event list" \
                  % (event.id, event.maxchan, event.t, event.template.id)
            return
        self.tree.Delete(event.itemID)
        template = event.template
        del template.events[event.id] # del event from its template's event dict
        self.session.events[event.id] = event # restore event to unsorted session.events
        template.update_wave() # update mean template waveform
        event.template = None # unbind event's template from event
        event.itemID = None # no longer applicable
        data = [event.id, event.maxchan, event.t]
        self.list.InsertRow(0, data) # stick it at the top of the list, is there a better place to put it?
        # TODO: maybe re-sort the list

    def MoveCurrentEvents2Template(self, which='selected'):
        if which == 'selected':
            template = self.GetFirstSelectedTemplate()
        elif which == 'new':
            template = None # indicates we want a new template
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        for row in selected_rows:
            event = self.listRow2Event(row)
            if event.wave.data != None: # only move it to template if it's got wave data
                template = self.MoveEvent2Template(event, row, template) # if template was None, it isn't any more
            else:
                print "can't add event %d to template because its data isn't accessible" % event.id
        if template != None and template.plot != None: # if it exists and it's plotted
            self.UpdateObjectsInPlot([template]) # update its plot

    def MoveCurrentObjects2List(self):
        for itemID in self.tree.GetSelections():
            if itemID: # check if event's tree parent (template) has already been deleted
                obj = self.tree.GetItemPyData(itemID)
                if obj.__class__ == Event:
                    self.MoveEvent2List(obj)
                elif obj.__class__ == Template:
                    self.DeleteTemplate(obj)
        self.OnTreeSelectChanged() # update plot

    def MoveCurrentEvents2Trash(self):
        """Move currently selected events in event list to trash"""
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        for row in selected_rows:
            event = self.listRow2Event(row)
            self.MoveEvent2Trash(event, row)

    def GetFirstSelectedTemplate(self):
        for itemID in self.tree.GetSelections():
            obj = self.tree.GetItemPyData(itemID)
            if obj.__class__ == Template:
                return obj
            # no template selected, check to see if an event is selected in the tree, grab its template
            elif obj.__class__ == Event:
                return obj.template
        return None
