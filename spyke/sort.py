"""Spike sorting classes and frame"""

__authors__ = 'Martin Spacek, Reza Lotun'

import os
import wx

import numpy as np

from spyke.gui import wxglade_gui


class Session(object):
    """A spike sorting session, in which you can do multiple Detection runs,
    build Templates up from events in those Detection runs, and then use Templates
    to sort spike Events.
    Formerly known as a Collection.
    A .sort file is a single sort Session object, pickled and gzipped"""
    def __init__(self, detector=None, srffname=None, probe=None):
        self.detector = detector # this session's current Detector object
        self.srffname = os.path.basename(srffname) # last srf file that was open in this session
        self.probe = probe # only one probe design per session allowed
        self.detections = [] # history of detection runs, in chrono order
        self.events = {} # all events detected in this sort session across all Detection runs, indexed by unique ID
        self.templates = {} # first hierarchy of templates

        self._detid = 0 # used to count off unqiue Detection run IDs
        self._eventid = 0 # used to count off unique Event IDs
        self._templid = 0 # used to count off unique Template IDs

    def set_stream(self, stream=None):
        """Set Stream object for self's detector and all detections,
        for unpickling purposes"""
        self.detector.set_stream(stream)
        for detection in self.detections:
            detection.detector.set_stream(stream)

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
        self.chans = None # chans enabled
        self.events = {} # member spike events that make up this template
        #self.surffname # not here, let's allow templates to have spikes from different files?

    def mean(self):
        """Returns mean waveform from member spikes"""
        for spike in self.spikes:
            pass
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
        #self.session # optional attrib, if this is an unsorted spike?
        # or, instead of .session and .template, just make a .parent attrib?
        #self.srffname # originating surf file name, with path relative to self.session.datapath
        #self.chans # necessary? see .template
        #self.cluster = None # cluster run this Spike was sorted on
        #self.rip = None # rip this Spike was sorted on

    def __getitem__(self, key):
        """Return WaveForm for this event given slice key"""
        assert key.__class__ == slice
        stream = self.detection.detector.stream
        if stream != None: # stream is available
            self.wave = stream[key] # let stream handle the slicing, save result
            return self.wave
        else: # stream unavailable, assume .wave was set before last pickling
            return self.wave[key] # slice existing .wave

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        del d['itemID'] # remove tree item ID, since that'll have changed anyway on unpickle
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
        self.deselect_all = False
        self.deselect_count = 0

        columnlabels = ['ID', 'chan', 'time'] # event list column labels
        for coli, label in enumerate(columnlabels):
            self.list.InsertColumn(coli, label)
        for coli in range(len(columnlabels)): # this needs to be in a separate loop it seems
            self.list.SetColumnWidth(coli, wx.LIST_AUTOSIZE_USEHEADER) # resize columns to fit

        self.root = self.tree.AddRoot('Templates')

        self.list.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def get_session(self):
        return self.spykeframe.session

    def set_session(self):
        raise RunTimeError, "SortFrame's .session not setable"

    session = property(get_session, set_session) # make this a property for proper behaviour after unpickling

    def OnClose(self, event):
        frametype = self.__class__.__name__.lower().replace('frame', '') # remove 'Frame' from class name
        self.spykeframe.HideFrame(frametype)

    def OnLeftDown(self, evt):
        """TODO: clicked item id may not be the same as spike event id, check with non consecutive spike event ids in list ctrl"""
        i, pos = self.list.HitTest(evt.GetPosition())
        print 'left down, on item %d' % i
        #selected = (sf.list.GetItemState(11, wx.LIST_STATE_SELECTED) == wx.LIST_STATE_SELECTED)
        #if not selected:
        #    self.list.Select(i)
        ctrl = evt.ControlDown()
        if not ctrl: # plain left click any item, deselect all
            self.deselect_all = True
            self.deselect_count = self.list.GetSelectedItemCount() # skip this many deselect events
        evt.Skip() # let select/deselect event processing happen

    def OnListSelect(self, evt):
        """Item selection event in list control"""
        # TODO: maybe use GetData instead, assign event id integer, so no conversion from str necessary
        eventi = int(evt.GetText()) # seems to always return the item's 0th column, which is its Event ID
        event = self.session.events[eventi]
        self.spikesortpanel.add_event(event)
        self.chartsortpanel.add_event(event)

    def OnListDeselect(self, evt):
        """Item deselection event in list control"""
        #nselected = self.list.GetSelectedItemCount()
        #print 'nselected: %d' % nselected
        eventi = int(evt.GetText()) # seems to always return the item's 0th column, which is its Event ID
        #self._events_to_remove.append(eventi)
        if self.deselect_all: # fast removal of all events
            self.spikesortpanel.remove_all_events()
            self.chartsortpanel.remove_all_events()
            self.deselect_all = False
            self.deselect_count -= 1
        elif self.deselect_count > 0: # wait for all the deselect events to be processed
            self.deselect_count -= 1
        else: # deselect just the one event
            event = self.session.events[eventi]
            self.spikesortpanel.remove_event(event)
            self.chartsortpanel.remove_event(event)
    '''
    def OnListBeginDrag(self, evt):
        """Begin list drag event"""
        pass
    '''
    def OnListKeyDown(self, evt):
        """Event list control key down event"""
        key = evt.GetKeyCode()
        if key in [ord('A'), wx.WXK_LEFT]: # wx.WXK_RETURN doesn't seem to work
            self.MoveCurrentEvents2Template(which='selected')
        elif key in [ord('C'), ord('N'), ord('T')]: # wx.WXK_SPACE doesn't seem to work
            self.MoveCurrentEvents2Template(which='new')
        if key in [wx.WXK_LEFT, wx.WXK_RIGHT]:
            evt.Veto() # stop propagation as navigation event or something

    def MoveCurrentEvents2Template(self, which='selected'):
        selected_rows = self.list_GetSelections()
        if which == 'selected':
            template = self.GetFirstSelectedTemplate()
        elif which == 'new':
            template = None # indicates we want a new template
        for row in selected_rows:
            event = self.row2event(row)
            template = self.MoveEvent2Template(event, row, template)

    def OnTreeKeyDown(self, evt):
        key = evt.GetKeyCode()
        if key in [wx.WXK_DELETE, wx.WXK_RIGHT]:
            self.MoveCurrentEvents2List()

    def MoveCurrentEvents2List(self):
        selected_itemIDs = self.tree.GetSelections()
        for itemID in selected_itemIDs:
            obj = self.tree.GetItemPyData(itemID)
            if obj.__class__ == Event:
                self.MoveEvent2List(obj)

    def row2event(self, row):
        eventi = int(self.list.GetItemText(row))
        event = self.session.events[eventi]
        return event

    def list_GetSelections(self):
        """Stupid wxPython list ctrl doesn't have this as a method,
        should really subclass ListCtrl and add this to it.
        Returns row indices of selected list items"""
        selected_rows = []
        first = self.list.GetFirstSelected()
        if first == -1:
            return selected_rows
        selected_rows.append(first)
        last = first
        while True:
            next = self.list.GetNextSelected(last)
            if next == -1:
                return selected_rows
            selected_rows.append(next)
            last = next

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

    def MoveEvent2Template(self, event, row, template=None):
        """Move a spike event from list control row to a template in the tree.
        If template is None, create a new one"""
        self.list.DeleteItem(row) # remove it from the event list
        self.list.Select(row) # automatically select the new item at that position
        if template == None:
            template = self.CreateTemplate()
        template.events[event.id] = event # add event to template
        event.template = template # bind template to event
        event.itemID = self.tree.AppendItem(template.itemID, 'e'+str(event.id)) # add event to tree
        self.tree.SetItemPyData(event.itemID, event) # associate event tree item with event
        self.tree.Expand(template.itemID) # expand template
        return template

    def MoveEvent2List(self, event):
        """Move a spike event from a template in the tree back to the list control"""
        self.tree.Delete(event.itemID)
        del event.template.events[event.id] # del event from its template's event dict
        del event.template # unbind event's template from event
        del event.itemID # no longer applicable
        data = [event.id, event.chan, event.t]
        self.list_InsertRow(0, data)
        #self.list.Append(row)

    def list_InsertRow(self, row, data):
        """wxPython can be dumb.
        data is a list of strings or numbers, one per column"""
        row = self.list.InsertStringItem(row, str(data[0])) # inserts data's first column
        for coli, val in enumerate(data[1:]):
            self.list.SetStringItem(row, coli+1, str(val))

    def CreateTemplate(self):
        """Create, select, and return a new template"""
        template = Template(self.session, self.session._templid, parent=None)
        self.session._templid += 1 # inc for next unique Template
        self.session.templates[template.id] = template # add template to session
        template.itemID = self.tree.AppendItem(self.root, 't'+str(template.id)) # add template to tree
        self.tree.SetItemPyData(template.itemID, template) # associate template tree item with template
        self.tree.Expand(self.root) # make sure root is expanded
        self.tree.UnselectAll() # first unselect all items in tree
        self.tree.SelectItem(template.itemID) # now select the newly created template
        return template


'''
####################################################

class SpikeTreeCtrl(wx.TreeCtrl):
    def __init__(self, layout, *args, **kwargs):
        wx.TreeCtrl.__init__(self, *args, **kwargs)
        self.layout = layout

    def OnCompareItems(self, item1, item2):
        data1 = self.GetItemPyData(item1)
        data2 = self.GetItemPyData(item2)

        rank1 = self.layout[data1.channel][1] # y coord
        rank2 = self.layout[data2.channel][1] # y coord
        return cmp(rank1, rank2) * -1         # reverse

class TemplateTreeCtrl(wx.TreeCtrl):
    pass

class SpikeSorter(wx.Frame):
    def __init__(self, parent, id, title, layout, fname, collection=None, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.fname = fname or 'collection.pickle'   # name of serialized obj
        self.collection = collection
        self.recycleBinNode = None
        self.currSelected = None
        self.currentlyPlotted = {}  # keep track of currently selected
        self.layout = layout        # to order spikes appropriately

        self.SetTitle('Spike Sorter')

        # set up our tree controls
        self.makeTrees()

        self.root_Templates = self.tree_Templates.AddRoot('Templates')
        self.root_Spikes = self.tree_Spikes.AddRoot('Spikes')
        self.roots = (self.root_Templates, self.root_Spikes)
        self.trees = (self.tree_Templates, self.tree_Spikes)

        # initial template of spikes we want to sort
        self.setData(self.collection)

        # make sure the two trees are expanded
        for tree, root in zip(self.trees, self.roots):
            tree.Expand(root)
            tree.Unselect()
            tree.SelectItem(root, select=True)

        # bind our events
        self.registerEvents()

        # lay everything out
        self.layoutControls()

        # TODO: drag and drop functionality - it isn't ready yet!
        #dt = TreeTemplateDropTarget(self.tree_Templates,
        #                     self.root_Templates,
        #                     self.collection)
        #self.tree_Templates.SetDropTarget(dt)
        #
        #dt = TreeSpikeDropTarget(self.tree_Spikes,
        #                  self.root_Spikes,
        #                         self.collection)
        #self.tree_Spikes.SetDropTarget(dt)

    def makeTrees(self):
        arg_dict = {}
        arg_dict['style'] = wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE | \
                            wx.SUNKEN_BORDER | \
                            wx.TR_EXTENDED | wx.TR_SINGLE #| wx.TR_HIDE_ROOT

        self.tree_Templates = TemplateTreeCtrl(self, -1, **arg_dict)
        self.tree_Spikes = SpikeTreeCtrl(self.layout, self, -1, **arg_dict)


    def layoutControls(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer = wx.GridSizer(1, 2, 0, 0)
        grid_sizer.Add(self.tree_Templates, 1, wx.EXPAND, 0)
        grid_sizer.Add(self.tree_Spikes, 1, wx.EXPAND, 0)
        sizer.Add(grid_sizer, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        #sizer_1.Fit(self)
        self.Layout()

    def setData(self, collection):
        """ Display our collection data. The general use case is a session
        of spike sorting within our collection. This would arise out of two
        main scenarios:
            1) We have generated a set of unordered spikes. This represents
               a fresh collection we'd like to sort manually.
            2) We're presented a collection of (partially) ordered spikes
               (either via some automated process or an earler manual sorting)
               that we'd like to further sort.
        """
        # The right pane displays the unordered spikes
        for spike in self.collection.unsorted_spikes:
            item = self.tree_Spikes.AppendItem(self.root_Spikes, str(spike))
            self.tree_Spikes.SetPyData(item, spike)

        # sort all the unordered spikes to spatial order
        self.tree_Spikes.SortChildren(self.root_Spikes)

        # restore recycle bin
        try:
            self.collection.recycle_bin
            if self.collection.recycle_bin:
                rbin = self.tree_Spikes.AppendItem(self.root_Spikes, 'Recycle Bin')
                self.recycleBinNode = rbin
                for spike in self.collection.recycle_bin:
                    item = self.tree_Spikes.AppendItem(rbin, str(spike))
                    self.tree_Spikes.SetPyData(item, spike)
        except AttributeError:
            # this is for backwards compatibility
            pass

        # The left pane represents our currently (sorted) templates
        for template in self.collection:
            item = self.tree_Templates.AppendItem(self.root_Templates, str(template))
            self.tree_Templates.SetPyData(item, template)

            # add all the spikes within the templates
            for spike in template:
                sp_item = self.tree_Templates.AppendItem(item, str(spike))
                self.tree_Templates.SetPyData(sp_item, spike)

            self.tree_Templates.Expand(item)

    def registerEvents(self):
        """Binds all of the events to the spike and template tree controls"""
        for tree in self.trees:
            # Node activation and transition
            #self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.onActivate, tree)
            self.Bind(wx.EVT_TREE_SEL_CHANGING, self.onSelChanging, tree)
            self.Bind(wx.EVT_TREE_SEL_CHANGED, self.onSelChanged, tree)
            self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.onRightClick, tree)

            # tree collapsing
            self.Bind(wx.EVT_TREE_ITEM_COLLAPSING, self.onCollapsing, tree)

            # keyboard interaction
            self.Bind(wx.EVT_TREE_KEY_DOWN, self.onKeyDown, tree)

            # node deletion
            #self.Bind(wx.EVT_TREE_DELETE_ITEM, self.onDelete, tree)

            # Mouse drag and drop
            #self.Bind(wx.EVT_TREE_BEGIN_DRAG, self.onBeginDrag, tree)
            #self.Bind(wx.EVT_TREE_END_DRAG, self.onBeginDrag, tree)
            #self.Bind(wx.EVT_TREE_BEGIN_RDRAG, self.onBeginDrag, tree)

            # Nodel label editing
            #self.Bind(wx.EVT_TREE_BEGIN_LABEL_EDIT, self.beginEdit, tree)
            #self.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.endEdit, tree)


    def onSelChanging(self, evt):
        """Remove currently plotted"""
        if self.currSelected:
            tree = self.FindFocus()
            self._cmdPlot(evt, tree, self.currSelected, False)
            self.currSelected = None

    def onSelChanged(self, evt):
        if not self.currSelected:
            it = evt.GetItem()
            tree = self.FindFocus()
            if it.IsOk():
                self.currSelected = it
                self._cmdPlot(evt, tree, it, True)

    def vetoOnRoot(handler):
        """Decorator which vetoes a certain event if it occurs on a root node"""
        def new_handler(obj, evt):
            it = evt.GetItem()
            if it in obj.roots:
                evt.Veto()
                return
            return handler(obj, evt)
        return new_handler

    def onKeyDown(self, evt):
        key_event = evt.GetKeyEvent()
        code = key_event.GetKeyCode()
        point = key_event.GetPosition()
        tree = self.FindFocus()
        self.currentTree = tree
        it = tree.GetSelection()

        # dummy function
        nothing = lambda *args: None
        # XXX : 'j' == wx.WXK_J?
        keyCommands = {
                        wx.WXK_RETURN   : self._modifyPlot,
                        wx.WXK_SPACE    : self._modifyPlot,
                        #wx.WXK_UP       : self._selectPrevItem,
                        wx.WXK_LEFT     : nothing,
                        wx.WXK_RIGHT    : nothing,
                        #wx.WXK_DOWN     : self._selectNextItem,
                        wx.WXK_TAB      : self._toggleTreeFocus,
                        ord('j')        : self._selectNextItem,
                        ord('J')        : self._selectNextItem,
                        ord('k')        : self._selectPrevItem,
                        ord('K')        : self._selectPrevItem,
                        ord('h')        : self._selectParent,
                        ord('H')        : self._selectParent,
                        ord('l')        : self._selectFirstChild,
                        ord('L')        : self._selectFirstChild,
                        ord('t')        : self._createTemplate,
                        ord('T')        : self._createTemplate,
                        ord('a')        : self._addToTemplate,
                        ord('A')        : self._addToTemplate,
                        ord('d')        : self._deleteSpike,
                        ord('D')        : self._deleteSpike,
                        wx.WXK_DELETE   : self._deleteSpike,
                        ord('s')        : self._serialize,
                        ord('S')        : self._serialize,
                    }

        for cmd, action in keyCommands.iteritems():
            if code == cmd:
                action(evt, tree, it)
                break

        evt.Skip()

    def _serialize(self, evt, *args):
        """Serialize our collection"""
        evt = evt.GetKeyEvent()
        #if not evt.ControlDown():
        #    return

        print '\n*************  Saving to ', self.fname, '  ************\n'
        write_collection(self.collection, self.fname)
        print 'Done.'
        self.currentTree.SelectItem(args[1])

    # XXX can use code generation to auto-make these _select* methods
    def _selectNextItem(self, evt, currTree, it):
        ni = self.currentTree.GetNextSibling(it)
        if ni.IsOk():
            self.currentTree.SelectItem(ni)

    def _selectPrevItem(self, evt, currTree, it):
        pi = self.currentTree.GetPrevSibling(it)
        if pi.IsOk():
            self.currentTree.SelectItem(pi)

    def _selectParent(self, evt, currTree, it):
        """go to parent"""
        par = self.currentTree.GetItemParent(it)
        if par.IsOk():
            self.currentTree.SelectItem(par)

    def _selectFirstChild(self, evt, currTree, it):
        chil, cookie = self.currentTree.GetFirstChild(it)
        if chil.IsOk():
            self.currentTree.SelectItem(chil)

    def _toggleTreeFocus(self, evt, currTree, it=None):
        """Toggles focus between the two trees and returns the newly focused-upon tree"""
        self.onSelChanging(evt)
        for tr in self.trees:
            if tr != currTree:
                tr.SetFocus()
                break
        self.currentTree = tr
        #self.onSelChanged(evt)
        if not self.currSelected:
            #it = evt.GetItem()
            tree = self.FindFocus()
            it = tree.GetSelection()
            if it.IsOk():
                self.currSelected = it
                self._cmdPlot(evt, tree, it, True)
        return tr

    #XXX merge the following two methods
    def _cmdPlot(self, evt, tree, item, visible=True):
        if item in self.roots:
            return
        event = PlotEvent(myEVT_PLOT, self.GetId())
        data = tree.GetPyData(item)
        event.channels = [True] * len(self.layout)
        if self._isTemplate(item):
            event.channels = data.active_channels
            data = data.mean()
        if visible:
            event.plot = data
            event.top = True
        else:
            event.remove = data
        event.colour = 'y'
        self.GetEventHandler().ProcessEvent(event)

    # XXX: merge
    def clickedChannel(self, channels):
        curr = self.tree_Templates.GetSelection()
        self.redisplayChannels(curr, channels)

    def redisplayChannels(self, template_Node, channels):
        event = PlotEvent(myEVT_PLOT, self.GetId())
        data = self.tree_Templates.GetPyData(template_Node)
        if self._isTemplate(template_Node):
            template_plot = data.mean()
            event.plot = template_plot

            # toggle active channels in template for selected channel
            for i, chan in enumerate(data.active_channels):
                if channels[i]:
                    data.active_channels[i] = not chan
            event.channels = data.active_channels
            event.colour = 'y'

            self.GetEventHandler().ProcessEvent(event)

    def _modifyPlot(self, evt, tree, item):

        if item in self.roots:
            return

        if item == self.recycleBinNode:
            return

        event = PlotEvent(myEVT_PLOT, self.GetId())
        data = self.currentTree.GetPyData(item)

        event.channels = [True] * len(self.layout)
        #event.isTemplate = self._isTemplate(item)
        # we're plotting a template
        if self._isTemplate(item):
            event.channels = data.active_channels
            colour = 'r'
            data = data.mean()
        else:
            colour = 'g'
        if not tree.IsBold(item):
            self.currentTree.SetItemBold(item)
            event.plot = data
        else:
            self.currentTree.SetItemBold(item, False)
            event.remove = data
        event.colour = colour
        self.GetEventHandler().ProcessEvent(event)

    # XXX: change name
    def _deleteSpike(self, evt, tree, it):
        """Delete spike..."""
        def isTemplatePlotted(templateNode):
            permplot = self.tree_Templates.IsBold(templateNode)
            currplot = it == templateNode
            return permplot or currplot

        def isSpikePlotted(node):
            return self.tree_Spikes.IsBold(node)

        if it in self.roots:
            return

        if it == self.recycleBinNode:
            return


        if tree == self.tree_Templates:
            # We have two cases
            # CASE 1: we're deleting a spike. If so, we have to remove
            # the spike and place it in the spike list. We then also have
            # to check if the current template is empty, at which point it
            # needs to be removed
            if not self._isTemplate(it):
                templateNode = tree.GetItemParent(it)
                template = tree.GetPyData(templateNode)
                count = tree.GetChildrenCount(templateNode)

                isPlotted = isTemplatePlotted(templateNode)
                if isPlotted:
                    self._modifyPlot(evt, tree, templateNode)

                spike = tree.GetPyData(it)

                # remove it from its original collection
                self._moveSpike(tree, it)
                self._copySpikeNode(tree, self.root_Spikes, it)

                self._removeCurrentSelection(it, tree)

                if count - 1 == 0:
                    # we're deleting the last spike in this template
                    self.collection.templates.remove(template)
                    tree.Delete(templateNode)
                    return

                # replot
                if isPlotted:
                    self._modifyPlot(evt, tree, templateNode)

            # CASE 2: we're deleting a template. In this case we should move
            # all the spikes within the template to the spike list
            else:
                if isTemplatePlotted(it):
                    #self._modifyPlot(evt, tree, it)
                    self._cmdPlot(evt, tree, it, False)

                child, cookie = tree.GetFirstChild(it)
                template = tree.GetPyData(it)

                # move all the children to the spike pane
                if child.IsOk():
                    self._moveSpike(tree, child)
                    self._copySpikeNode(tree, self.root_Spikes, child)
                    while True:
                        nextitem, cookie = tree.GetNextChild(it, cookie)
                        if not nextitem.IsOk():
                            break
                        self._moveSpike(tree, nextitem)
                        self._copySpikeNode(tree, self.root_Spikes, nextitem)
                    tree.DeleteChildren(it)
                    tree.Delete(it)
                    self.collection.templates.remove(template)

        if tree == self.tree_Spikes:
            if not self.recycleBinNode:
                self.recycleBinNode = self.tree_Spikes.AppendItem(self.root_Spikes,
                                                                'Recycle Bin')
            if isSpikePlotted(it):
               self._modifyPlot(evt, tree, it)
            self._moveSpike(tree, it)
            self._copySpikeNode(tree, self.recycleBinNode, it)
            self._removeCurrentSelection(it, tree)
            self.tree_Spikes.Collapse(self.recycleBinNode)

    def _copySpikeNode(self, source_tree, parent_node, it):
        """Copy spike node it from source_tree to parent_node, transferring
        state (such as currently plotted) as well"""
        # get info for copied spike
        data = source_tree.GetPyData(it)
        text = source_tree.GetItemText(it)

        for tree in self.trees:
            if not tree == source_tree:
                dest_tree = tree

        if dest_tree == self.tree_Spikes and self.recycleBinNode:
        # new spike node
            ns = dest_tree.InsertItemBefore(parent_node, self.recycleBinNode,
                    text)
        else:
            ns = dest_tree.AppendItem(parent_node, text)

        dest_tree.SetPyData(ns, data)
        bold = dest_tree.IsBold(it)
        dest_tree.SetItemBold(ns, bold)

        return ns

    def _moveSpike(self, src_tree, src_node, dest_template=None):
        """Used to manage collection data structure"""
        # get the actual spike
        spike = src_tree.GetPyData(src_node)


        if src_tree == self.tree_Spikes:
            if dest_template is None:
                # we're going to the recycle bin
                self.collection.unsorted_spikes.remove(spike)
                self.collection.recycle_bin.append(spike)
            else:
                # there IS a dest template, so update the destination template
                dest_template.add(spike)

        else:
            # we're dealing with the template tree
            # remove spike from its original template
            for template in self.collection.templates:
                if spike in template:
                    template.remove(spike)
                    break

            # we're moving a spike OUT of a template. Thus we should
            # add it to unsorted_spikes
            self.collection.unsorted_spikes.append(spike)


    def _isTemplate(self, item):
        par = self.tree_Templates.GetItemParent(item)
        return par == self.root_Templates

    def onlyOn(permitree):
        """Will create a decorator which only permits actions on permitree"""
        def decor_func(handler):
            def new_handler(obj, evt, tree, it):
                if not tree == getattr(obj, permitree):
                    return
                return handler(obj, evt, tree, it)
            return new_handler
        return decor_func

    # XXX
    def _removeFromTemplate(tree, it):
        pass

    def _removeCurrentSelection(self, it, tree=None):
        # XXX hackish
        if tree is None:
            tree = self.tree_Spikes
        # make sure we select the next spike, so we don't jump to root
        ni = tree.GetNextSibling(it)
        if ni.IsOk():
            tree.SelectItem(ni)
            tree.Delete(it)
        else:
            # we're the last item in the list - select previous item
            pi = tree.GetPrevSibling(it)
            if pi.IsOk():
                tree.SelectItem(pi)
                tree.Delete(it)
            else:
                # we're the last item of all - in that case select the root
                tree.SelectItem(self.root_Spikes)
                tree.Delete(it)

    @onlyOn('tree_Spikes')
    def _addToTemplate(self, evt, tree, it):
        """ Add selected item to the currently selected template. """

        # the semantics of 'a' are as follows:
        #   1) In the spike tree, 'a' on the root nodes does nothing
        #   2) In the spike tree, 'a' on any other node (a spike) works
        #      differently depending on what area of the template tree
        #      is highlighted
        #          i) The Root - create new template
        #         ii) A template - add to that currently selected template
        #        iii) A spike - add to the same template (i.e. make a child
        #             of the parent template of spike.

        def isTemplatePlotted(templateNode):
            return self.tree_Templates.IsBold(templateNode)
        # check if item is the spike root - do nothing
        if it == self.root_Spikes:
            return

        # get the currently selected template
        curr = self.tree_Templates.GetSelection()

        # check if curr is a template, otherwise, it's a spike so get
        # it's parent template
        if self._isTemplate(curr):
            dest = curr
        elif curr == self.root_Templates:
            return self._createTemplate(evt, tree, it)
        else:
            dest = self.tree_Templates.GetItemParent(curr)

                #templateNode = tree.GetItemParent(it)
                #template = tree.GetPyData(templateNode)

        isPlotted = isTemplatePlotted(dest)
        if isPlotted:
            self._modifyPlot(evt, self.tree_Templates, dest)
        # get the template we're going to add to
        template = self.tree_Templates.GetPyData(dest)

        # copy spike to this template
        ns = self._copySpikeNode(tree, dest, it)

        # make sure template is expanded and new spike selected
        self.tree_Templates.Expand(curr)
        self.tree_Templates.SelectItem(ns)

        # move spike to template
        self._moveSpike(tree, it, template)

        self._removeCurrentSelection(it)
        if isPlotted:
            self._modifyPlot(evt, self.tree_Templates, dest)

        # make sure the deselected channels aren't shown
        #template = self.tree_Templates.GetPyData(dest)
        #print template.active_channels
        #self.redisplayChannels(dest, [not x for x in template.active_channels])


        print 'Collection: '
        print str(self.collection)


    @onlyOn('tree_Spikes')
    def _createTemplate(self, evt, tree, it):
        # check if item is the spike root - do nothing
        if it == self.root_Spikes:
            return

        # create new template and update our collection
        new_template = Template()
        self.collection.templates.append(new_template)

        # create new template node
        nt = self.tree_Templates.AppendItem(self.root_Templates, 'Template')

        # copy spike
        ns = self._copySpikeNode(tree, nt, it)

        # make sure template is expanded and new spike selected
        self.tree_Templates.Expand(nt)
        self.tree_Templates.SelectItem(ns)


        # move spike to template
        self._moveSpike(tree, it, new_template)

        # set the data for the new template node
        self.tree_Templates.SetPyData(nt, new_template)

        #XXX
        self._removeCurrentSelection(it)


        print 'Collection: '
        print str(self.collection)

    @vetoOnRoot
    def onRightClick(self, evt):
        it = evt.GetItem()
        point = evt.GetPoint()
        tree = self._getTreeId(point)
        self._modifyPlot(evt, tree, it)
        tree.SelectItem(it)

    def endEdit(self, evt):
        # change the name of the spike/template
        new_label = evt.GetLabel()
        if not new_label:
            evt.Veto()
            return
        item = evt.GetItem()
        tree = self._getTreeId(item)
        data = tree.GetPyData(item)
        data.name = new_label
        tree.SetPyData(item, data)

    @vetoOnRoot
    def beginEdit(self, evt):
        pass

    @vetoOnRoot
    def onCollapsing(self, evt):
        """ Called just before a node is collapsed. """
        pass

    @vetoOnRoot
    def onBeginDrag(self, evt):
        # consider a single node drag for now
        tree = self._getTreeId(evt.GetPoint())
        it = evt.GetItem()

        # get info
        data = tree.GetPyData(it)
        text = tree.GetItemText(it)

        # package up data and state
        dragged = DraggedSpike()
        dragged.spike = data
        dragged.bold = tree.IsBold(it)
        spike_drag = wx.CustomDataObject(wx.CustomDataFormat('spike'))
        spike_drag.SetData(cPickle.dumps(dragged, 1))

        spike_source = wx.DropSource(tree)
        spike_source.text = text
        spike_source.SetData(spike_drag)

        # XXX indicate that current node is undergoing a transition

        # this is BLOCKED until drop is either blocked or accepted
        # wx.DragCancel
        # wx.DragCopy
        # wx.DragMove
        # wx.DragNone
        self.dragged = it
        #evt.Allow()
        res = spike_source.DoDragDrop(wx.Drag_AllowMove)
        res = 1

        if res & wx.DragCancel:
            ### XXX: do something more?
            return

        if res & wx.DragMove:
            #tree.Delete(it)
            return

    @vetoOnRoot
    def OnEndDrag(self, event):
        if not event.GetItem().IsOk():
            return

        try:
            old = self.dragged
        except:
            return

        tree = self._getTreeId(event.GetPoint())
        it = event.GetItem()
        parent = tree.GetItemParent(it)
        if not parent.IsOk():
            return

        tree.Delete(old)
        tree.InsertItem(parent, it)

    def _getTreeId(self, point):
        """ Get the tree id that item is under - this is useful since this
        widget is comprised of two trees.
        """
        hittest_flags = set([wx.TREE_HITTEST_ONITEM,
                             wx.TREE_HITTEST_ONITEMBUTTON,
                             wx.TREE_HITTEST_ONITEMICON,
                             wx.TREE_HITTEST_ONITEMINDENT,
                             wx.TREE_HITTEST_ONITEMLABEL,
                             wx.TREE_HITTEST_ONITEMRIGHT])
        # HIT TEST
        for tree in self.trees:
            sel_item, flags = tree.HitTest(point)
            for flag in hittest_flags:
                if flag & flags:
                    return tree

        raise Exception('Tree not found??!!')


class SpikeDropSource(wx.DropSource):
    def __init__(self, tree):
        pass

class DraggedSpike(object):
    """ Represents the dragged data. We need to store the actual data itself
    and its state in the tree - namely whether it's bold or not.
    """
    def __init__(self):
        self.spike = None
        self.bold = None


class TreeDropTarget(wx.PyDropTarget):
    def __init__(self, tree, root, collection):
        wx.DropTarget.__init__(self)
        self.tree = tree
        self.root = root
        self.collection = collection
        self.df = wx.CustomDataFormat('spike')
        self.cdo = wx.CustomDataObject(self.df)
        self.SetDataObject(self.cdo)

        self.new_item = None
        self.new_coords = None

        flags = (wx.TREE_HITTEST_ONITEM,
                 wx.TREE_HITTEST_ONITEMBUTTON,
                 wx.TREE_HITTEST_ONITEMICON,
                 wx.TREE_HITTEST_ONITEMINDENT,
                 wx.TREE_HITTEST_ONITEMLABEL,
                 wx.TREE_HITTEST_ONITEMRIGHT,
                 wx.TREE_HITTEST_ONITEMUPPERPART,
                 wx.TREE_HITTEST_ONITEMSTATEICON,
                 wx.TREE_HITTEST_ONITEMLOWERPART)
        self.hittest_flags = 0
        for f in flags:
            self.hittest_flags = self.hittest_flags | f

    def mouseOnItem(self, hflag):
        if hflag & self.hittest_flags:
            return True
        return False

    def setTempItem(self, x, y, prev_item):
        pass


class TreeTemplateDropTarget(TreeDropTarget):
    """Logic behind dragging and dropping onto list of templates"""
    def __init__(self, *args, **kwargs):
        TreeDropTarget.__init__(self, *args, **kwargs)
        self.new_template = None

    def OnDragOver(self, x, y, default):
        sel_item, flags = self.tree.HitTest((x, y))
        if self.mouseOnItem(flags):
            # check if we should create a new *template*
            # first, check if we're the last child of our parent, and check if
            # our parent is the last child of the root
            par = self.tree.GetItemParent(sel_item)
            if self.tree.GetLastChild(par) == sel_item:
                # we're the last child
                if self.tree.GetLastChild(self.root) == par:
                    # we're in the last template
                    self.createNewTemplate(x, y, flags, sel_item)
                    #self.setTempItem(x, y, flags, self.new_template)


            # we have to check if the item we're hovering over is
            # 1) A template item. If so, we have to expand the template to
            #    reveal the spikes contained within in it and enter mode 2
            # 2) A spike item within a template. If so, we have to add a new
            #    spike after it.
            if self.tree.GetItemParent(sel_item) == self.root:
                # we're over a template - make sure we expand
                self.tree.Expand(sel_item)
            else:
                # we're *within* a template
                self.setTempItem(x, y, flags, sel_item)

        return default

    def createNewTemplate(self, x, y, flags, sel_item):
        self.new_template = self.tree.AppendItem(self.root, 'New Template')
        self.deleteTempItem()
        self.new_template_child = self.tree.AppendItem(self.new_template, 'New Spike')
        #self.new_item = self.tree.AppendItem(self.new_template, 'New Spike')
        #self.new_coords

    def deleteTempItem(self):
        if self.new_item:
            self.tree.Delete(self.new_item)
            self.new_item = None
            self.new_coords = None

    def deleteTemplate(self):
        if self.new_template:
            self.tree.Delete(self.new_template_child)
            self.tree.Delete(self.new_template)
            self.new_template = None

    def setTempItem(self, x, y, flags, sel_item):
        def createItem():
            #if self.tree.GetLastChild(self.root) == sel_item:
            #    # we're
            #if self.tree.GetItemParent(sel_item) == self.root:
            #    # we're over a template - make sure we expand
            #    self.tree.Expand(sel_item)
            #    self.new_item = self.tree.AppendItem(sel_item, 'new spike')
            #else:

            template = self.tree.GetItemParent(sel_item)
            self.new_item = self.tree.InsertItem(template, sel_item, 'new spike')
            self.new_coords = (x, y)


        if not self.new_item:
            createItem()

        if self.new_item:
            it_x, it_y = self.new_coords
            upper = it_y - 5
            lower = it_y + 20
            if y <= upper or y >= lower:
                self.deleteTempItem()
                self.deleteTemplate()
                createItem()

    def OnData(self, x, y, default):
        if self.GetData():
            data = cPickle.loads(self.cdo.GetData())
            self.tree.SetItemText(self.new_item, data.name)
            self.tree.SetPyData(self.new_item, data)
            self.tree.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
            self.tree.SelectItem(self.new_item)
            self.new_item = None
            self.new_coords = None
        else:
            return wx.DragCancel


class TreeSpikeDropTarget(TreeDropTarget):
    """Logic behind dragging and dropping onto list of spikes"""

    def OnDragOver(self, x, y, default):
        # when we begin dragging, we have left our original item with
        # the mouse still down. The user is now hunting for the spot in
        # which to dropped the dragged data. At this time instant, we should
        # first check where we are
        sel_item, flags = self.tree.HitTest((x, y))

        if flags & wx.TREE_HITTEST_NOWHERE:
            return

        if self.mouseOnItem(flags):
            #self.setTempItem(x, y, flags, sel_item)
            if self.new_item:
                self.tree.SetItemDropHighlight(self.new_item, False)
            self.new_item = sel_item
            self.tree.SetItemDropHighlight(sel_item)
            self.new_coords = (x, y)

        return default

    def OnData(self, x, y, default):
        sel_item, flags = self.tree.HitTest((x, y))
        if flags & wx.TREE_HITTEST_NOWHERE:
            # we dropping on nothing revoke all our actions
            if self.new_item:
                #self.tree.Delete(self.new_item)
                self.new_item = None
                self.new_coords = None
            return wx.DragCancel

        if self.GetData():
            dragged_data = cPickle.loads(self.cdo.GetData())
            data = dragged_data.spike
            self.tree.SetItemText(self.new_item, data.name)
            self.tree.SetPyData(self.new_item, data)
            self.tree.SetItemBold(self.new_item, dragged_data.bold)
            self.tree.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
            self.tree.SelectItem(self.new_item)
            self.tree.SetItemDropHighlight(self.new_item, False)
            self.new_item = None
            self.new_coords = None
            return default
        else:
            return wx.DragCancel

    def setTempItem(self, x, y, flags, sel_item):
        print x, y

        def createItem():
            self.new_item = self.tree.InsertItem(self.root, sel_item, 'spike')
            self.new_coords = (x, y)
            self.tree.SelectItem(self.new_item)

        if self.new_item:
            #it_x, it_y = self.new_coords
            #upper = it_y - 5
            #lower = it_y + 20
            #if y <= upper or y >= lower:
            self.tree.Delete(self.new_item)
            self.new_item = None
            self.new_coords = None
            createItem()

        if not self.new_item:
            createItem()


#####----- Tests

from spyke.gui.plot import Opener

class SorterWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.op = op

        self.plotPanel = ClickableSortPanel(self, self.op.layout.SiteLoc)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass

class TestApp(wx.App):
    def __init__(self, fname, *args, **kwargs):
        self.fname = fname
        # XXX - turn of redirection of stdout to wx display window
        kwargs['redirect'] = False
        wx.App.__init__(self, *args, **kwargs)

    def OnInit(self):
        op = Opener()
        self.op = op
        if self.fname:
            col = load_collection(self.fname)
        else:
            col = self.makeCol()

        self.sorter = SpikeSorter(None, -1, 'Spike Sorter', op.layout.SiteLoc, self.fname, col, size=(500, 600))
        self.plotter = SorterWin(None, -1, 'Plot Sorter', op, size=(200, 900))
        self.SetTopWindow(self.sorter)
        self.sorter.Show(True)
        self.plotter.Show(True)

        self.Bind(EVT_PLOT, self.handlePlot, self.sorter)
        self.Bind(EVT_CLICKED_CHANNEL, self.handleClickedChannel, self.plotter.plotPanel)
        return True

    def handlePlot(self, evt):
        if evt.plot:
            self.plotter.plotPanel.add(evt.plot, evt.colour, evt.top, evt.channels)
        elif evt.remove:
            self.plotter.plotPanel.remove(evt.remove, evt.colour)

    def handleClickedChannel(self, evt):
        self.sorter.clickedChannel(evt.selected_channels)

    def makeCol(self):
        from spyke.stream import WaveForm
        from random import randint
        #simp = BipolarAmplitudeFixedThresh(self.op.dstream, self.op.dstream.records[0].TimeStamp)
        simp = DynamicMultiPhasic(self.op.dstream, self.op.dstream.records[0].TimeStamp)
        spikes = []
        for i, spike in enumerate(simp):
            spikes.append(spike)
            if i > 500:
                break
        col = Collection()

        #for i in range(10):
        #    col.unsorted_spikes.append(Spike(WaveForm(), channel=i, event_time=randint(1, 10000)))
        col.unsorted_spikes = spikes
        return col


if __name__ == "__main__":
    import sys

    fname = None
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    app = TestApp(fname)
    app.MainLoop()
'''
