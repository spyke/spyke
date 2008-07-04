"""Spike sorting classes and frame"""

__authors__ = 'Martin Spacek, Reza Lotun'

import os
import wx

import numpy as np

from spyke.core import WaveForm
from spyke.gui import wxglade_gui
from spyke.gui.plot import DEFEVENTTW

# save all Event waveforms, even for those that have never been plotted or added to a template
SAVEALLEVENTWAVES = False


class Session(object):
    """A spike sorting session, in which you can do multiple Detection runs,
    build Templates up from events in those Detection runs, and then use Templates
    to sort spike Events.
    Formerly known as a Collection.
    A .sort file is a single sort Session object, pickled and gzipped"""
    def __init__(self, detector=None, srffname=None, probe=None, stream=None):
        self.detector = detector # this session's current Detector object
        self.srffname = os.path.basename(srffname) # last srf file that was open in this session
        self.probe = probe # only one probe design per session allowed
        self.detections = [] # history of detection runs, in chrono order, reuse deleted Detection IDs
        self.stream = stream
        self.events = {} # all events detected in this sort session across all Detection runs, indexed by unique ID
        self.templates = {} # first hierarchy of templates

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
        """Append events to self
        TODO: ensure you don't have duplicate events from previous detection runs"""
        #for e in events.values():
        #    self.events[e.id] = e
        self.events.update(events)


class Detection(object):
    """A spike detection run, which happens every time Search is pressed.
    When you're merely searching for the previous/next spike with
    F2/F3, that's not considered a detection run"""
    def __init__(self, session, detector, id=None, datetime=None, events_array=None):
        self.session = session # parent sort Session
        self.detector = detector # Detector object used in this Detection run
        self.id = id
        self.datetime = datetime
        self.events_array = events_array # unsorted spike events, 2D array output of Detector.search
        #self.spikes = {} # a dict of Event objects? a place to note which events in this detection have been chosen as either member spikes of a template or sorted spikes. Need this here so we know which Spike objects to delete from this sort Session when we delete a Detection
        self.trash = {} # discarded events

    def set_events(self):
        """Convert .events_array to dict of Event objects, inc session's _eventid counter"""
        self.events = {}
        for t, chan in self.events_array.T: # same as iterate over cols of non-transposed events array
            e = Event(self.session._eventid, chan, t, self)
            self.session._eventid += 1 # inc for next unique Event
            self.events[e.id] = e

    def __eq__(self, other):
        """Compare detection runs by their .events_array"""
        # TODO: see if there's any overlap between self.events and other.events, and raise a warning in a dialog box or something
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
        self.wave = None
        #self.surffname # not here, let's allow templates to have spikes from different files?

    def update_wave(self):
        """Update mean waveform, should call this every time .events is modified.
        Setting .events as a property to do so automatically doesn't work, because
        properties only catch name binding events, not modification to an object
        that's already been bound"""
        if self.events == {}: # no member spikes yet
            self.wave = None
            return
        wave = self.wave or WaveForm()
        data = []
        relts = np.arange(self.trange[0], self.trange[1], self.session.tres) # timestamps relative to spike time
        event = self.events.values()[0] # grab a random event
        if event.wave == None:
            event.update_wave(trange=self.trange)
        sampfreq = wave.sampfreq or event.wave.sampfreq
        chan2i = wave.chan2i or event.wave.chan2i
        for event in self.events.values():
            # check each event for a .wave
            if event.wave == None or not ((event.wave.ts - event.t) == relts).all():
                event.update_wave(trange=self.trange)
            assert event.wave.sampfreq == sampfreq # being really thorough here...
            assert event.wave.chan2i == chan2i
            data.append(event.wave.data)
        data = np.asarray(data).mean(axis=0)
        # TODO: search data and find maxchan, set self.maxchan
        wave.data = data
        wave.ts = relts
        wave.sampfreq = sampfreq
        wave.chan2i = chan2i
        self.wave = wave
        return self.wave

    def get_trange(self):
        return self._trange

    def set_trange(self, trange=(-DEFEVENTTW/2, DEFEVENTTW/2)):
        """Reset self's time range relative to t=0 spike time,
        update slice of member spikes, and mean waveform"""
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
    '''
    def pop(self, spikeid):
        return self.spikes.pop(spikeid)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        del d['itemID'] # remove tree item ID, since that'll have changed anyway on unpickle
        return d


class Event(object):
    """Either an unsorted event, or a member spike in a Template,
    or a sorted spike in a Detection (or should that be sort Session?)"""
    def __init__(self, id, chan, t, detection):
        self.id = id # some integer for easy user identification
        self.chan = chan # necessary? see .template
        self.t = t # timestamp, waveform is centered on this?
        self.detection = detection # Detection run self was detected on
        self.template = None # template object it belongs to, None means self is an unsorted event
        self.wave = None # WaveForm
        self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        #self.session # optional attrib, if this is an unsorted spike?
        # or, instead of .session and .template, just make a .parent attrib?
        #self.srffname # originating surf file name, with path relative to self.session.datapath
        #self.chans # necessary? see .template
        #self.cluster = None # cluster run this Spike was sorted on
        #self.rip = None # rip this Spike was sorted on
        # try loading it right away on init, instead of waiting until plot, see if it's fast enough
        # nah, too slow after doing an OnSearch, don't load til plot or til Save (ie pickling)
        #self.update_wave()

    def update_wave(self, trange=(-DEFEVENTTW/2, DEFEVENTTW/2)):
        """Load/update self's waveform, defaults to default event time window centered on self.t"""
        self.wave = self[self.t+trange[0] : self.t+trange[1]]
        return self.wave

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
        if SAVEALLEVENTWAVES and self.wave == None:
            # make sure .wave is loaded before pickling to file
            self.update_wave()
        d = self.__dict__.copy()
        # clear tree item ID in dict, since that'll have changed anyway on unpickle
        # TODO: this might be dangerous, cuz we rely on itemID in OnTreeRightDown
        d['itemID'] = None
        return d


class Cluster(object):
    """Cluster is an object that holds all the settings of a
    cluster run. A cluster run is when you compare each of the
    detected but unsorted spikes in the sort Session to all templates,
    and decide which template it best fits. Compare with a Rip"""

    def match(self, spike):
        pass

# or just have two options in the Sort pane: Rip against: detected events; entire file

class Rip(object):
    """Holds all the Rip settings. A rip is when you take each template and
    slide it across the entire file. A spike is detected and
    sorted at timepoints where the error between template and file falls below
    some threshold"""
    pass


class ClusterRip(Cluster, Rip):
    """A hybrid of the two. Rip each template across all of the unsorted spikes
    instead of across the entire file"""
    pass


class SortFrame(wxglade_gui.SortFrame):
    """Sort frame"""
    def __init__(self, *args, **kwargs):
        wxglade_gui.SortFrame.__init__(self, *args, **kwargs)
        self.spykeframe = self.Parent

        self.listTimer = wx.Timer(owner=self.list)

        self.lastSelectedListEvents = []
        self.lastSelectedTreeEvents = []
        self.lastSelectedTreeTemplates = []

        columnlabels = ['eID', 'chan', 'time'] # event list column labels
        for coli, label in enumerate(columnlabels):
            self.list.InsertColumn(coli, label)
        for coli in range(len(columnlabels)): # this needs to be in a separate loop it seems
            self.list.SetColumnWidth(coli, wx.LIST_AUTOSIZE_USEHEADER) # resize columns to fit

        self.list.Bind(wx.EVT_TIMER, self.OnListTimer)
        self.list.Bind(wx.EVT_RIGHT_DOWN, self.OnListRightDown)
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
        raise RunTimeError, "SortFrame's .session not setable"

    session = property(get_session, set_session) # make this a property for proper behaviour after unpickling

    def OnSize(self, evt):
        """Put code here to re-save reflines_background"""
        print 'resizing'
        evt.Skip()

    def OnClose(self, evt):
        frametype = self.__class__.__name__.lower().replace('frame', '') # remove 'Frame' from class name
        self.spykeframe.HideFrame(frametype)
    '''
    TODO: on any selection event, start or restart a timer for say 0.1 sec, record the item selected/deselected, and append it to a sel/desel queu. Then, when the timer eventually runs down to zero and fires a timing event, read the whole sel/desel queue, OR it with what was selected previously, and execute the approriate minimum number of clear/plot actions all at once. This prevents unnecessary draws/redraws/clears. Should make plotting and unplotting faster, and flicker free. Also gets rid of need for any mouse click event handling, like OnLeftDown, and associated key events that haven't even been written yet.
        - event better: when timer runs down, just execute self.list.GetSelections and compare to previous list of selections, and execute your plots accordingly. This way, you don't even need to build up a queue on each sel/desel event. This will make all the sel event handling even faster, and allow you to reduce the timer duration for faster response.
        - after completion of each selection epoch (say 0.1s), save current buffer as new background?
    '''

    def OnListSelect(self, evt):
        """Restart list selection timer"""
        self.listTimer.Stop()
        self.listTimer.Start(milliseconds=1, oneShot=True) # only fire one timer event after specified interval

    def OnListDeselect(self, evt):
        self.OnListSelect(evt)

    def OnListTimer(self, evt):
        """Run when started timer runs out and triggers a TimerEvent"""
        selectedRows = self.list.GetSelections()
        selectedListEvents = [ self.listRow2Event(row) for row in selectedRows ]
        removeEvents = [ sel for sel in self.lastSelectedListEvents if sel not in selectedListEvents ]
        addEvents = [ sel for sel in selectedListEvents if sel not in self.lastSelectedListEvents ]

        #import cProfile
        #cProfile.runctx('self.spikesortpanel.removeEvents(removeEvents)', globals(), locals())
        #cProfile.runctx('self.spikesortpanel.addEvents(addEvents)', globals(), locals())

        print 'events to remove: %r' % [ event.id for event in removeEvents ]
        self.spikesortpanel.removeEvents(removeEvents)
        #self.chartsortpanel.removeEvents(removeEvents)
        print 'events to add: %r' % [ event.id for event in addEvents ]
        self.spikesortpanel.addEvents(addEvents)
        #self.chartsortpanel.addEvents(addEvents)
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
            self.spikesortpanel.event_plots[event.id] # is it plotted?
            selected = True # if so, item must be selected
            print 'event %d in event_plots' % event.id
        except KeyError:
            selected = False # item is not selected
            print 'event %d not in event_plots' % event.id
        self.list.Select(itemID, on=not selected) # toggle selection, this fires sel event, which updates the plot

    def OnListKeyDown(self, evt):
        """Event list control key down event"""
        key = evt.GetKeyCode()
        if key in [ord('A'), wx.WXK_LEFT]: # wx.WXK_RETURN doesn't seem to work
            self.MoveCurrentEvents2Template(which='selected')
        elif key in [ord('C'), ord('N'), ord('T')]: # wx.WXK_SPACE doesn't seem to work
            self.MoveCurrentEvents2Template(which='new')
        if key in [wx.WXK_LEFT, wx.WXK_RIGHT]:
            evt.Veto() # stop propagation as navigation event or something

    def OnTreeSelectChanged(self, evt=None):
        """Due to bugs #2307 and #626, a SEL_CHANGED event isn't fired when
        deselecting the currently focused item in a tree with the wx.TR_MULTIPLE
        flag set, as it is here"""
        print 'in OnTreeSelectChanged'
        selected_itemIDs = self.tree.GetSelections()
        selectedTreeEvents = []
        selectedTreeTemplates = []
        for itemID in selected_itemIDs:
            item = self.tree.GetItemPyData(itemID)
            if item.__class__ == Event:
                selectedTreeEvents.append(item)
            elif item.__class__ == Template:
                selectedTreeTemplates.append(item)
            else:
                raise ValueError, 'weird type of item selected'
        removeEvents = [ sel for sel in self.lastSelectedTreeEvents if sel not in selectedTreeEvents ]
        removeTemplates = [ sel for sel in self.lastSelectedTreeTemplates if sel not in selectedTreeTemplates ]
        addEvents = [ sel for sel in selectedTreeEvents if sel not in self.lastSelectedTreeEvents ]
        addTemplates = [ sel for sel in selectedTreeTemplates if sel not in self.lastSelectedTreeTemplates ]

        #import cProfile
        #cProfile.runctx('self.spikesortpanel.removeEvents(removeEvents)', globals(), locals())
        #cProfile.runctx('self.spikesortpanel.addEvents(addEvents)', globals(), locals())

        self.RemoveEventsFromPlot(removeEvents)
        self.RemoveTemplatesFromPlot(removeTemplates)
        self.AddEvents2Plot(addEvents)
        self.AddTemplates2Plot(addTemplates)
        self.lastSelectedTreeEvents = selectedTreeEvents # save for next time
        self.lastSelectedTreeTemplates = selectedTreeTemplates # save for next time

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
        # this would be nice, but doesn't work cuz apparently somehow the
        # selection TreeEvent happens before the MouseEvent that caused it:
        #selected = not self.tree.IsSelected(itemID)
        # here is a yucky workaround:
        obj = self.tree.GetItemPyData(itemID) # either an Event or a Template
        if obj.__class__ == Event:
            plots = self.spikesortpanel.event_plots
        elif obj.__class__ == Template:
            plots = self.spikesortpanel.template_plots
        try:
            plots[obj.id] # is it plotted?
            selected = True # if so, item must be selected
            print 'obj %d is in its plots list' % obj.id
        except KeyError:
            selected = False # item is not selected
            print 'obj %d is not in its plots list' % obj.id
        self.tree.SelectItem(itemID, select=not selected) # toggle
        # restore selection of previously selected events and templates that were
        # inadvertently deselected earlier in the TreeEvent
        for plottedEventi in self.spikesortpanel.event_plots.keys():
            plottedEvent = self.session.events[plottedEventi]
            if plottedEvent.itemID != obj.itemID: # if it's not the one whose selected state we just handled
                self.tree.SelectItem(plottedEvent.itemID) # enforce its selection
        for plottedTemplatei in self.spikesortpanel.template_plots.keys():
            plottedTemplate = self.session.templates[plottedTemplatei]
            if plottedTemplate.itemID != obj.itemID: # if it's not the one whose selected state we just handled
                self.tree.SelectItem(plottedTemplate.itemID) # enforce its selection
        # now plot accordingly
        self.OnTreeSelectChanged()
        #evt.Veto() # not defined for mouse event?
        #evt.StopPropagation() # doesn't seem to do anything

    def OnTreeKeyDown(self, evt):
        key = evt.GetKeyCode()
        #print 'key down: %r' % key
        if key in [wx.WXK_DELETE, ord('D')]:
            self.MoveCurrentEvents2List()
        elif key in [wx.WXK_UP, wx.WXK_DOWN]: # keyboard selection hack around multiselect bug
            wx.CallAfter(self.OnTreeSelectChanged)
        evt.Skip()

    def OnTreeKeyUp(self, evt):
        key = evt.GetKeyCode()
        #print 'key up: %r' % key
        if key == wx.WXK_SPACE: # space only triggered on key up, see bug #4448
            wx.CallAfter(self.OnTreeSelectChanged)
        evt.Skip()

    def Append2EventList(self, events):
        """Append events to self's event list control"""
        SiteLoc = self.spykeframe.session.probe.SiteLoc
        for e in events.values():
            row = [str(e.id), e.chan, e.t]
            self.list.Append(row)
            ycoord = SiteLoc[e.chan][1]
            # add ycoord of maxchan of event as data for this row, use item
            # count instead of counting from 0 cuz you want to handle there
            # already being items in the list from prior append/removal
            self.list.SetItemData(self.list.GetItemCount()-1, ycoord)
        self.list.SortItems(cmp) # sort the list by maxchan from top to bottom of probe
        #width = wx.LIST_AUTOSIZE_USEHEADER # resize columns to fit
        # hard code column widths for precise control, autosize seems buggy
        for coli, width in {0:40, 1:40, 2:80}.items(): # (eID, chan, time)
            self.list.SetColumnWidth(coli, width)

    def AddEvents2Plot(self, events):
        print 'events to add: %r' % [ event.id for event in events ]
        self.spikesortpanel.addEvents(events)
        #self.chartsortpanel.addEvents(events)

    def RemoveEventsFromPlot(self, events):
        print 'events to remove: %r' % [ event.id for event in events ]
        self.spikesortpanel.removeEvents(events)
        #self.chartsortpanel.removeEvents(events)

    def AddTemplates2Plot(self, templates):
        print 'templates to add: %r' % [ template.id for template in templates ]
        self.spikesortpanel.addTemplates(templates)
        #self.chartsortpanel.addTemplates(templates)

    def RemoveTemplatesFromPlot(self, templates):
        print 'templates to remove: %r' % [ template.id for template in templates ]
        self.spikesortpanel.removeTemplates(templates)
        #self.chartsortpanel.removeTemplates(templates)

    #TODO: should self.OnTreeSelectChanged() (update plot) be called more often at the end of many of the following methods?:

    def CreateTemplate(self):
        """Create, select, and return a new template"""
        template = Template(self.session, self.session._templid, parent=None)
        self.session._templid += 1 # inc for next unique Template
        self.session.templates[template.id] = template # add template to session
        root = self.tree.GetRootItem()
        if not root.IsOk(): # if tree doesn't have a valid root item
            root = self.tree.AddRoot('Templates')
        template.itemID = self.tree.AppendItem(root, 't'+str(template.id)) # add template to tree
        self.tree.SetItemPyData(template.itemID, template) # associate template tree item with template
        #self.tree.Expand(root) # make sure root is expanded
        self.tree.UnselectAll() # first unselect all items in tree
        self.tree.SelectItem(template.itemID) # now select the newly created template
        return template

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
        """Move a spike event from list control row to a template in the tree.
        If template is None, create a new one"""
        self.list.DeleteItem(row) # remove it from the event list
        self.list.Select(row) # automatically select the new item at that position
        if template == None:
            template = self.CreateTemplate()
        template.events[event.id] = event # add event to template
        template.update_wave() # update mean template waveform
        event.template = template # bind template to event
        event.itemID = self.tree.AppendItem(template.itemID, 'e'+str(event.id)) # add event to tree
        self.tree.SetItemPyData(event.itemID, event) # associate event tree item with event
        self.tree.Expand(template.itemID) # expand template
        return template

    def MoveEvent2List(self, event):
        """Move a spike event from a template in the tree back to the list control"""
        self.tree.Delete(event.itemID)
        template = event.template
        del template.events[event.id] # del event from its template's event dict
        template.update_wave() # update mean template waveform
        event.template = None # unbind event's template from event
        event.itemID = None # no longer applicable
        data = [event.id, event.chan, event.t]
        self.list.InsertRow(0, data)

    def MoveCurrentEvents2Template(self, which='selected'):
        selected_rows = self.list.GetSelections()
        if which == 'selected':
            template = self.GetFirstSelectedTemplate()
        elif which == 'new':
            template = None # indicates we want a new template
        for row in selected_rows:
            event = self.listRow2Event(row)
            template = self.MoveEvent2Template(event, row, template)

    def MoveCurrentEvents2List(self):
        selected_itemIDs = self.tree.GetSelections()
        for itemID in selected_itemIDs:
            obj = self.tree.GetItemPyData(itemID)
            if obj.__class__ == Event:
                event = obj
                template = event.template
                self.MoveEvent2List(event)
                if len(template.events) == 0: # if this template doesn't have any events left in it
                    self.DeleteTemplate(template) # delete it
            elif obj.__class__ == Template:
                template = obj
                self.DeleteTemplate(template)
        self.OnTreeSelectChanged() # update plot

    def GetFirstSelectedTemplate(self):
        selected_itemIDs = self.tree.GetSelections()
        for itemID in selected_itemIDs:
            obj = self.tree.GetItemPyData(itemID)
            if obj.__class__ == Template:
                return obj
        # no template selected, check to see if an event is selected in the tree, grab its template
        for itemID in selected_itemIDs:
            obj = self.tree.GetItemPyData(itemID)
            if obj.__class__ == Event:
                return obj.template
        return None
