
import wx

from spyke.detect import Spike, Template, Collection

tree1 = [
        "Event 1",
        ["Event 2", [
            "Spike 1",
            "Spike 2",
            "Spike 3",
            ]],
        "Event 3",
        ]

tree2 = [
        "Spike 235",
        "Spike 73",
        "Spike 23",
        ]


class SpikeDropSource(wx.DropSource):
    def __init__(self, tree):
        pass

class SpikeSorter(wx.Frame):
    def __init__(self, parent, id, title, templates=None, spikes=None, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)

        # set up our tree controls
        self.setUpTrees()

        # initial template of spikes we want to sort
        self.setData(spikes, templates)

        #self.AddTreeNodes(self.tree_Templates, self.templateRoot, tree1)
        #self.AddTreeNodes(self.tree_Spikes, self.spikeRoot, tree2)

        for tree, root in zip(self.trees, self.roots):
            tree.Expand(root)

        self.registerEvents()

        self.__set_properties()
        self.__do_layout()

    def setUpTrees(self):
        # keep references to our trees and roots
        self.tree_Templates, self.tree_Spikes = None, None
        self.templateRoot, self.spikeRoot = None, None
        self.makeTrees()
        self.roots = (self.templateRoot, self.spikeRoot)
        self.trees = (self.tree_Templates, self.tree_Spikes)

    def setData(self, templates, spikes):
        """ Set up our template data structures. We'd like to be able to
        support two use cases:
            1) We're passed in a collection of spikes that we want to manually
               organize into a collection of templates.
            2) We're passed in a a collection of templates that we want to
               refine.
        """
        self.spikes = spikes or Template()
        self.templates = templates or Collection()

        if not spikes.__class__.__name__  == Spike.__name__:
            raise Exception()   # XXX: clean this up
        if not templates.__class__.__name__ == Template.__name__:
            raise Exception()   # XXX
        # attached to each root will be a spike template. Informally
        # the tree on the right is a temporary holding window for unsorted
        # spikes, and the tree on the left denotes either 'final' templates
        # that are being built from the spikes on the right, or a set of
        # existing templates that are to be refined and or/viewed




    def registerEvents(self):
        for tree in self.trees:
            wx.EVT_TREE_BEGIN_DRAG(tree, tree.GetId(), self.onBeginDrag)
            #wx.EVT_TREE_BEGIN_RDRAG(tree, tree.GetId(), self.testDrag)
            #wx.EVT_TREE_SEL_CHANGING(tree, tree.GetId(), self.maintain)
            wx.EVT_TREE_ITEM_ACTIVATED(tree, tree.GetId(), self.onActivate)
            wx.EVT_TREE_ITEM_COLLAPSING(tree, tree.GetId(), self.onCollapsing)
            wx.EVT_TREE_BEGIN_LABEL_EDIT(tree, tree.GetId(), self.beginEdit)

    def _evtRootVeto(self, evt):
        """ Veto an event if it happens to a root. """
        it = evt.GetItem()
        if it in self.roots:
            evt.Veto()
            return True
        return False

    def beginEdit(self, evt):
        self._evtRootVeto(evt)

    def onCollapsing(self, evt):
        """ Called just before a node is collapsed. """
        self._evtRootVeto(evt)

    def onBeginDrag(self, evt):
        pass

    def onActivate(self, evt):
        pass

    def makeTrees(self):
        tempKwds, spikeKwds = {}, {}
        tempKwds['style'] = wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE | \
                           wx.SUNKEN_BORDER | wx.TR_EDIT_LABELS | \
                           wx.TR_EXTENDED | wx.TR_MULTIPLE #| wx.TR_HIDE_ROOT
        self.tree_Templates = wx.TreeCtrl(self, -1, **tempKwds)

        spikeKwds['style'] = wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE | \
                           wx.SUNKEN_BORDER | wx.TR_EDIT_LABELS | \
                           wx.TR_EXTENDED | wx.TR_MULTIPLE #| wx.TR_HIDE_ROOT
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

class TestApp(wx.App):
    def OnInit(self):
        sorter = SpikeSorter(None, -1, 'Spike Sorter', None, size=(500, 600))
        self.SetTopWindow(sorter)
        sorter.Show(True)
        return True

if __name__ == "__main__":
    app = TestApp()
    app.MainLoop()

