
import cPickle
import wx

import spyke
from spyke.layout import *
from spyke.detect import Spike, Template, Collection, SimpleThreshold
from spyke.gui.plot import SortPanel


class PlotEvent(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.plot = None
        self.remove = None

myEVT_PLOT = wx.NewEventType()
EVT_PLOT = wx.PyEventBinder(myEVT_PLOT, 1)

class TestApp(wx.App):
    def __init__(self, *args, **kwargs):
        #if not kwargs:
        #    kwargs = {}
        #kwargs['redirect'] = True
        #kwargs['filename'] = 'debug_out'
        wx.App.__init__(self, *args, **kwargs)

    def OnInit(self):
        op = Opener()
        self.op = op
        col = self.makeCol()
        self.sorter = SpikeSorter(None, -1, 'Spike Sorter', col, size=(500, 600))
        self.plotter = SorterWin(None, -1, 'Plot Sorter', op, size=(200, 900))
        self.SetTopWindow(self.sorter)
        self.sorter.Show(True)
        self.plotter.Show(True)

        self.Bind(EVT_PLOT, self.handlePlot, self.sorter)
        return True

    def handlePlot(self, evt):
        if evt.plot:
            self.plotter.plotPanel.add(evt.plot)
        elif evt.remove:
            self.plotter.plotPanel.remove(evt.remove)

    def makeCol(self):
        from spyke.stream import WaveForm
        from random import randint
        simp = SimpleThreshold(self.op.dstream, self.op.dstream.records[0].TimeStamp)
        spikes = []
        for i, spike in enumerate(simp):
            spikes.append(spike)
            if i > 20:
                break
        col = Collection()
        temp = Template()
        temp.add(Spike(WaveForm(), channel=20, event_time=1337))
        col.templates.append(temp)

        #for i in range(10):
        #    col.unsorted_spikes.append(Spike(WaveForm(), channel=i, event_time=randint(1, 10000)))
        col.unsorted_spikes = spikes
        return col


from spyke.gui.plot import Opener

class SorterWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.op = op

        self.plotPanel = SortPanel(self, self.op.layout.SiteLoc)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass



class SpikeDropSource(wx.DropSource):
    def __init__(self, tree):
        pass


class SpikeSorter(wx.Frame):
    def __init__(self, parent, id, title, collection=None, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)

        self.collection = collection

        # set up our tree controls
        self.setUpTrees()

        # initial template of spikes we want to sort
        self.setData(self.collection)

        #self.AddTreeNodes(self.tree_Templates, self.templateRoot, tree1)
        #self.AddTreeNodes(self.tree_Spikes, self.spikeRoot, tree2)

        for tree, root in zip(self.trees, self.roots):
            tree.Expand(root)

        self.registerEvents()

        self.__set_properties()
        self.__do_layout()

        for tree, root in zip(self.trees, self.roots):
            tree.Unselect()
            # set drop target
            dt = TreeDropTarget(tree, root)
            tree.SetDropTarget(dt)


    def setUpTrees(self):
        # keep references to our trees and roots
        self.tree_Templates, self.tree_Spikes = None, None
        self.templateRoot, self.spikeRoot = None, None
        self.makeTrees()
        self.roots = (self.templateRoot, self.spikeRoot)
        self.trees = (self.tree_Templates, self.tree_Spikes)

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
            item = self.tree_Spikes.AppendItem(self.spikeRoot, spike.name)
            self.tree_Spikes.SetPyData(item, spike)

        # The left pane represents our currently (sorted) templates
        for template in self.collection:
            item = self.tree_Templates.AppendItem(self.templateRoot, template.name)

            # add all the spikes within the templates
            for spike in template:
                sp_item = self.tree_Templates.AppendItem(item, spike.name)
                self.tree_Templates.SetPyData(sp_item, spike)
            self.tree_Templates.Expand(item)

            self.tree_Templates.SetPyData(item, template)

    def registerEvents(self):
        for tree in self.trees:
            wx.EVT_TREE_BEGIN_DRAG(tree, tree.GetId(), self.onBeginDrag)
            #wx.EVT_TREE_BEGIN_RDRAG(tree, tree.GetId(), self.testDrag)
            #wx.EVT_TREE_SEL_CHANGING(tree, tree.GetId(), self.maintain)
            wx.EVT_TREE_ITEM_ACTIVATED(tree, tree.GetId(), self.onActivate)
            wx.EVT_TREE_ITEM_COLLAPSING(tree, tree.GetId(), self.onCollapsing)
            wx.EVT_TREE_BEGIN_LABEL_EDIT(tree, tree.GetId(), self.beginEdit)
            wx.EVT_TREE_END_LABEL_EDIT(tree, tree.GetId(), self.endEdit)
            wx.EVT_TREE_ITEM_RIGHT_CLICK(tree, tree.GetId(), self.onRightClick)


    def onRightClick(self, evt):
        if self._evtRootVeto(evt):
            return

        event = PlotEvent(myEVT_PLOT, self.GetId())
        it = evt.GetItem()

        print 'Event item: ', it
        point = evt.GetPoint()
        tree = self._getTreeId(point)
        #tree.ToggleItemSelection(it)
        data = tree.GetPyData(it)

        if not tree.IsBold(it):
            tree.SetItemBold(it)
            event.plot = data
        else:
            tree.SetItemBold(it, False)
            event.remove = data

        tree.UnselectAll()
        self.GetEventHandler().ProcessEvent(event)


    def _evtRootVeto(self, evt):
        """ Veto an event if it happens to a root. """
        it = evt.GetItem()
        if it in self.roots:
            evt.Veto()
            return True
        return False

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

    def beginEdit(self, evt):
        self._evtRootVeto(evt)

    def onCollapsing(self, evt):
        """ Called just before a node is collapsed. """
        self._evtRootVeto(evt)

    def onBeginDrag(self, evt):
        #print evt.__class__.__dict__
        #print self.tree.GetSelections()
        # consider a single node drag for now

        # don't drag the roots
        if self._evtRootVeto(evt):
            return

        tree = self._getTreeId(evt.GetPoint())
        it = evt.GetItem()


        # get info
        data = tree.GetPyData(it)
        text = tree.GetItemText(it)
        #iteminfo.append((text, data))

        spike_drag = wx.CustomDataObject(wx.CustomDataFormat('spike'))
        spike_drag.SetData(cPickle.dumps(data, 1))

        spike_source = wx.DropSource(tree)
        spike_source.SetData(spike_drag)

        # this is BLOCKED until drop is either blocked or accepted
        # wx.DragCancel
        # wx.DragCopy
        # wx.DragMove
        # wx.DragNone
        res = spike_source.DoDragDrop(True)

        #     if len(items) > 0:
        #         break
        evt.Allow()

    def _getTreeId(self, point):
        """ Get the tree id that item is under - this is useful since this widget
        is comprised of two trees.
        """

        hittest_flags = (wx.TREE_HITTEST_ONITEM,
                         wx.TREE_HITTEST_ONITEMBUTTON,
                         wx.TREE_HITTEST_ONITEMICON,
                         wx.TREE_HITTEST_ONITEMINDENT,
                         wx.TREE_HITTEST_ONITEMLABEL,
                         wx.TREE_HITTEST_ONITEMRIGHT)
        # HIT TEST
        for tree in self.trees:
            sel_item, flags = tree.HitTest(point)
            print sel_item, flags
            if flags in hittest_flags:
                return tree


        ####### end hit test
        #for tree in self.trees:
        #    print 'TREE selections: ', tree.GetSelections()
        #    print 'We are looking for this item: ', item
        #    if item in tree.GetSelections():
        #        return tree
        raise Exception('Tree not found??!!')
        

    def onActivate(self, evt):
        pass

    def makeTrees(self):
        tempKwds, spikeKwds = {}, {}
        tempKwds['style'] = wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE | \
                           wx.SUNKEN_BORDER | wx.TR_EDIT_LABELS | \
                           wx.TR_EXTENDED | wx.TR_SINGLE #| wx.TR_HIDE_ROOT
        self.tree_Templates = wx.TreeCtrl(self, -1, **tempKwds)

        spikeKwds['style'] = wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE | \
                           wx.SUNKEN_BORDER | wx.TR_EDIT_LABELS | \
                           wx.TR_EXTENDED | wx.TR_SINGLE #| wx.TR_HIDE_ROOT
        self.tree_Spikes = wx.TreeCtrl(self, -1, **spikeKwds)

        self.templateRoot = self.tree_Templates.AddRoot('Templates')
        self.spikeRoot = self.tree_Spikes.AddRoot('Spikes')

    def AddTreeNodes(self, tree, parentItem, items):
        """
        Recursively traverses the data structure, adding tree nodes to
        match it.
        """
        for item in items:
            if type(item) == str:
                tree.AppendItem(parentItem, item)
            else:
                newItem = tree.AppendItem(parentItem, item[0])
                self.AddTreeNodes(tree, newItem, item[1])

    def __set_properties(self):
        # begin wxGlade: MyFrame.__set_properties
        self.SetTitle('Spike Sorter')
        # end wxGlade

    def __do_layout(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer = wx.GridSizer(1, 2, 0, 0)
        grid_sizer.Add(self.tree_Templates, 1, wx.EXPAND, 0)
        grid_sizer.Add(self.tree_Spikes, 1, wx.EXPAND, 0)
        sizer.Add(grid_sizer, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        #sizer_1.Fit(self)
        self.Layout()

#class TreeDrop(wx.DropSource):
#    def __init__(self, trees):
#        # XXX
#        wx.DropSource.__init__(self, tree)
    
class TreeDropTarget(wx.DropTarget):
    def __init__(self, tree, root):
        wx.DropTarget.__init__(self)
        self.tree = tree
        self.root = root
        self.df = wx.CustomDataFormat('spike')
        self.cdo = wx.CustomDataObject(self.df)
        self.SetDataObject(self.cdo)
        self.new_template = None
        self.new_coords = None
        self.hittest_flags = (wx.TREE_HITTEST_ONITEM,
                         wx.TREE_HITTEST_ONITEMBUTTON,
                         wx.TREE_HITTEST_ONITEMICON,
                         wx.TREE_HITTEST_ONITEMINDENT,
                         wx.TREE_HITTEST_ONITEMLABEL,
                         wx.TREE_HITTEST_ONITEMRIGHT)

    def OnEnter(self, x, y, default):
        # figure out what tree we're in
        # cache it
        #hittest_flags = (wx.TREE_HITTEST_ONITEM,
        #                 wx.TREE_HITTEST_ONITEMBUTTON,
        #                 wx.TREE_HITTEST_ONITEMICON,
        #                 wx.TREE_HITTEST_ONITEMINDENT,
        #                 wx.TREE_HITTEST_ONITEMLABEL,
        #                 wx.TREE_HITTEST_ONITEMRIGHT)
        # HIT TEST
        #for tree in self.trees:
        #    sel_item, flags = tree.HitTest(point)
        #    print sel_item, flags
        #    if flags in hittest_flags:
        #        return tree
        print self.tree
        return default

    def OnLeave(self):
        # reset our cached tree
        #print self.tree
        pass

    #def OnDrop(self, x, y):
    #    for tree in self.trees:
    #        id, flag = tree.HitTest((x, y))
    #
    # XXX: do some stuff
    #    return True

    def OnDragOver(self, x, y, default):
        sel_item, flags = self.tree.HitTest((x, y))
        # is it off our list of templates? create a new one
        if flags == wx.TREE_HITTEST_NOWHERE:
            print 'temp!: ', self.new_template
            if not self.new_template:
                self.new_coords = (x, y)
                self.new_template = self.tree.AppendItem(self.root, 'temp')
        elif flags in self.hittest_flags:
            if (x, y) != self.new_coords and self.new_template:
                self.tree.Delete(self.new_template)
                self.new_template = None
                self.new_coords = None
        return default

    def onData(self, x, y, default):
        return default

if __name__ == "__main__":
    app = TestApp()
    app.MainLoop()

