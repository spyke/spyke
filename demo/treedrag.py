
import cPickle
import wx

import spyke
from spyke.layout import *
from spyke.detect import Spike, Template, Collection, SimpleThreshold
from spyke.gui.events import *
from spyke.gui.plot import SortPanel



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
            tree.SelectItem(root, select=True)


        dt = TreeTemplateDropTarget(self.tree_Templates,
                            self.templateRoot,
                            self.collection)
        self.tree_Templates.SetDropTarget(dt)

        dt = TreeSpikeDropTarget(self.tree_Spikes,
                                 self.spikeRoot,
                                 self.collection)
        self.tree_Spikes.SetDropTarget(dt)

    def setUpTrees(self):
        # keep references to our trees and roots
        self.tree_Templates, self.tree_Spikes = None, None
        self.templateRoot, self.spikeRoot = None, None
        self.makeTrees()
        self.roots = (self.templateRoot, self.spikeRoot)
        self.trees = (self.tree_Templates, self.tree_Spikes)

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
            #wx.EVT_TREE_ITEM_ACTIVATED(tree, tree.GetId(), self.onActivate)
            wx.EVT_TREE_ITEM_COLLAPSING(tree, tree.GetId(), self.onCollapsing)
            wx.EVT_TREE_BEGIN_LABEL_EDIT(tree, tree.GetId(), self.beginEdit)
            wx.EVT_TREE_END_LABEL_EDIT(tree, tree.GetId(), self.endEdit)
            wx.EVT_TREE_ITEM_RIGHT_CLICK(tree, tree.GetId(), self.onRightClick)

            wx.EVT_TREE_KEY_DOWN(tree, tree.GetId(), self.onKeyDown)

    def vetoOnRoot(handler):
        """ Decorator which vetoes a certain event if it occurs on
        a root node.
        """
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
        if code == wx.WXK_RETURN:       # if we hit the enter key
            point = key_event.GetPosition()
            it = evt.GetItem()
            tree = self.FindFocus()
            it = tree.GetSelection()
            self._modifyPlot(point, tree, it)

        if code == wx.WXK_UP:
            pass

    def _modifyPlot(self, point, tree, item):
        event = PlotEvent(myEVT_PLOT, self.GetId())
        data = tree.GetPyData(item)

        if not tree.IsBold(item):
            tree.SetItemBold(item)
            event.plot = data
        else:
            tree.SetItemBold(item, False)
            event.remove = data
        self.GetEventHandler().ProcessEvent(event)

    @vetoOnRoot
    def onRightClick(self, evt):
        it = evt.GetItem()
        point = evt.GetPoint()
        tree = self._getTreeId(point)
        self._modifyPlot(point, tree, it)
        tree.SelectItem(it)

        #tree.UnselectAll()

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
        res = spike_source.DoDragDrop(wx.Drag_AllowMove)

        print 'Res', res

        if res == wx.DragMove:
            tree.Delete(it)

        if res == wx.DragCancel:
            pass


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

    def onActivate(self, evt):
        pass



class DraggedSpike(object):
    """ Represents the dragged data. We need to store the actual data itself
    and its state in the tree - namely whether it's bold or not.
    """
    def __init__(self):
        self.spike = None
        self.bold = None


class TreeDropTarget(wx.DropTarget):
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
    """ Logic behind dragging and dropping onto list of templates """
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
    """ Logic behind dragging and dropping onto list of spikes """

    def OnDragOver(self, x, y, default):
        # when we begin dragging, we have left our original item with
        # the mouse still down. The user is now hunting for the spot in
        # which to dropped the dragged data. At this time instant, we should
        # first check where we are
        sel_item, flags = self.tree.HitTest((x, y))

        if flags & wx.TREE_HITTEST_NOWHERE:
            return

        # as we move our mouse pointer, candidate spike nodes should be
        # created under where our current mouse position is. As we leave
        # these candidate nodes should be removed. We must only have one
        # candidate node at one time

        # explicitly check if we are over the root. We can only add a
        # candidate node as a *child* of the root.
        if sel_item == self.root:
            if self.new_item:
                self.tree.Delete(self.new_item)
                self.new_item = None
                self.new_coords = None
            # create a new item that is a child of root
            self.new_item = self.tree.InsertItemBefore(self.root, 0, 'spike')
            self.new_coords = (x, y)
            self.tree.SelectItem(self.new_item)


        if self.mouseOnItem(flags):
            self.setTempItem(x, y, flags, sel_item)

        return default

    def OnData(self, x, y, default):
        sel_item, flags = self.tree.HitTest((x, y))
        if flags & wx.TREE_HITTEST_NOWHERE:
            # we dropping on nothing revoke all our actions
            if self.new_item:
                self.tree.Delete(self.new_item)
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


if __name__ == "__main__":
    app = TestApp()
    app.MainLoop()

