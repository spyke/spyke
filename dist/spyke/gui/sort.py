""" spyke.gui.sort

spyke gui elements for spike sorting
"""

__author__ = 'Reza Lotun'

import cPickle

import wx

import spyke
from spyke.layout import *
from spyke.detect import Spike, Template, Collection, SimpleThreshold
from spyke.gui.events import *
from spyke.gui.plot import SortPanel


class SpikeSorter(wx.Frame):
    def __init__(self, parent, id, title, fname, collection=None, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.fname = fname or 'collection.pickle'   # name of serialized obj
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

        self.garbageBin = None

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
            #self.Bind(wx.EVT_TREE_BEGIN_DRAG, self.onBeginDrag, tree)
            #wx.EVT_TREE_END_DRAG(tree, tree.GetId(), self.OnEndDrag)
            #wx.EVT_TREE_BEGIN_RDRAG(tree, tree.GetId(), self.testDrag)
            #wx.EVT_TREE_SEL_CHANGING(tree, tree.GetId(), self.maintain)
            #wx.EVT_TREE_ITEM_ACTIVATED(tree, tree.GetId(), self.onActivate)
            self.Bind(wx.EVT_TREE_ITEM_COLLAPSING, self.onCollapsing, tree)
            self.Bind(wx.EVT_TREE_BEGIN_LABEL_EDIT, self.beginEdit, tree)
            self.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.endEdit, tree)
            self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.onRightClick, tree)

            self.Bind(wx.EVT_TREE_KEY_DOWN, self.onKeyDown, tree)
            #self.Bind(wx.EVT_TREE_DELETE_ITEM, self.onDelete, tree)


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
        point = key_event.GetPosition()
        tree = self.FindFocus()
        self.currentTree = tree
        it = tree.GetSelection()

        # dummy function
        nothing = lambda *args: None

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
                        ord('s')        : self._serialize,
                        ord('S')        : self._serialize,
                    }

        for cmd, action in keyCommands.iteritems():
            if code == cmd:
                action(evt, tree, it)
                break

        evt.Skip()

    def _serialize(self, evt, *args):
        """ Serialize our collection """
        evt = evt.GetKeyEvent()
        if not evt.ControlDown():
            return

        print '\n*************  Saving to ', self.fname, '  ************\n'
        try:
            try:
                f = file(self.fname, 'w')
                # use highest pickle protocol available
                cPickle.dump(self.collection, f, -1)
                self.currentTree.SelectItem(args[1])
            except:
                # XXX
                raise
        finally:
            f.close()

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
        # go to parent
        par = self.currentTree.GetItemParent(it)
        if par.IsOk():
            self.currentTree.SelectItem(par)

    def _selectFirstChild(self, evt, currTree, it):
        chil, cookie = self.currentTree.GetFirstChild(it)
        if chil.IsOk():
            self.currentTree.SelectItem(chil)

    def _toggleTreeFocus(self, evt, currTree, it=None):
        """ Toggles focus between the two trees and returns the newly
        focused-upon tree.
        """
        for tr in self.trees:
            if tr != currTree:
                tr.SetFocus()
        self.currentTree = tr
        return tr

    def _modifyPlot(self, evt, tree, item):

        if item in self.roots:
            return

        if item == self.garbageBin:
            return

        event = PlotEvent(myEVT_PLOT, self.GetId())
        data = self.currentTree.GetPyData(item)

        event.isTemplate = self._isTemplate(item)

        if not tree.IsBold(item):
            self.currentTree.SetItemBold(item)
            event.plot = data
        else:
            self.currentTree.SetItemBold(item, False)
            event.remove = data
        self.GetEventHandler().ProcessEvent(event)

    # XXX: change name
    def _deleteSpike(self, evt, tree, it):
        """ Delete spike ... """
        def isTemplatePlotted(templateNode):
            return self.tree_Templates.IsBold(templateNode)

        if it in self.roots:
            return

        if it == self.garbageBin:
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
                self._copySpikeNode(tree, self.spikeRoot, it)
                
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
                    self._modifyPlot(evt, tree, it)

                child, cookie = tree.GetFirstChild(it)
                template = tree.GetPyData(it)

                # move all the children to the spike pane
                if child.IsOk():
                    self._moveSpike(tree, child)
                    self._copySpikeNode(tree, self.spikeRoot, child)
                    while True:
                        nextitem, cookie = tree.GetNextChild(it, cookie)
                        if not nextitem.IsOk():
                            break
                        self._moveSpike(tree, nextitem)
                        self._copySpikeNode(tree, self.spikeRoot, nextitem)
                    tree.DeleteChildren(it)
                    tree.Delete(it)
                    self.collection.templates.remove(template)

        if tree == self.tree_Spikes:
            if not self.garbageBin:
                self.garbageBin = self.tree_Spikes.AppendItem(self.spikeRoot, 
                                                                'Recycle Bin')
            self._copySpikeNode(tree, self.garbageBin, it)
            self._removeCurrentSelection(it, tree)
            self.tree_Spikes.Collapse(self.garbageBin)

    def _copySpikeNode(self, source_tree, parent_node, it):
        """ Copy spike node it from source_tree to parent_node, transferring
        state (such as currently plotted) as well.
        """
        # get info for copied spike
        data = source_tree.GetPyData(it)
        text = source_tree.GetItemText(it)

        for tree in self.trees:
            if not tree == source_tree:
                dest_tree = tree

        if dest_tree == self.tree_Spikes and self.garbageBin:
        # new spike node
            ns = dest_tree.InsertItemBefore(parent_node, self.garbageBin,
                    text)    
        else:
            ns = dest_tree.AppendItem(parent_node, text)

        dest_tree.SetPyData(ns, data)
        bold = dest_tree.IsBold(it)
        dest_tree.SetItemBold(ns, bold)

        return ns

    def _moveSpike(self, src_tree, src_node, dest_template=None):
        """ Used to manage collection data structure """
        # get the actual spike
        spike = src_tree.GetPyData(src_node)


        if src_tree == self.tree_Spikes:
            # we're moving a spike from the unsorted spikes
            self.collection.unsorted_spikes.remove(spike)
        else:
            # remove it from its original collection
            for template in self.collection.templates:
                if spike in template:
                    template.remove(spike)
                    break

        if not dest_template is None:
            # update the destination template
            dest_template.add(spike)
        else:
            # we're moving a spike OUT of a template. Thus we should
            # add it to unsorted_spikes
            self.collection.unsorted_spikes.append(spike)


    def _isTemplate(self, item):
        par = self.tree_Templates.GetItemParent(item)
        return par == self.templateRoot

    def onlyOn(permitree):
        """ Will create a decorator which only permits actions on permitree """
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
                tree.SelectItem(self.spikeRoot)
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

        # check if item is the spike root - do nothing
        if it == self.spikeRoot:
            return

        # get the currently selected template
        curr = self.tree_Templates.GetSelection()

        # check if curr is a template, otherwise, it's a spike so get
        # it's parent template
        if self._isTemplate(curr):
            dest = curr
        elif curr == self.templateRoot:
            return self._createTemplate(evt, tree, it)
        else:
            dest = self.tree_Templates.GetItemParent(curr)

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

        print 'Collection: '
        print str(self.collection)


    @onlyOn('tree_Spikes')
    def _createTemplate(self, evt, tree, it):
        # check if item is the spike root - do nothing
        if it == self.spikeRoot:
            return

        # create new template and update our collection
        new_template = Template()
        self.collection.templates.append(new_template)

        # create new template node
        nt = self.tree_Templates.AppendItem(self.templateRoot, 'Template')

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
        print 'Res', res

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

    def onActivate(self, evt):
        pass


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

        self.plotPanel = SortPanel(self, self.op.layout.SiteLoc)

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
            try:
                try:
                    f = file(self.fname)
                    col = cPickle.load(f)
                except:
                    # XXX do something clever here
                    raise
            finally:
                f.close()
        else:
            col = self.makeCol()
        self.sorter = SpikeSorter(None, -1, 'Spike Sorter', self.fname, col, size=(500, 600))
        self.plotter = SorterWin(None, -1, 'Plot Sorter', op, size=(200, 900))
        self.SetTopWindow(self.sorter)
        self.sorter.Show(True)
        self.plotter.Show(True)

        self.Bind(EVT_PLOT, self.handlePlot, self.sorter)
        return True

    def handlePlot(self, evt):
        if evt.plot:
            self.plotter.plotPanel.add(evt.plot, evt.isTemplate)
        elif evt.remove:
            self.plotter.plotPanel.remove(evt.remove, evt.isTemplate)

    def makeCol(self):
        from spyke.stream import WaveForm
        from random import randint
        simp = SimpleThreshold(self.op.dstream, self.op.dstream.records[0].TimeStamp)
        spikes = []
        for i, spike in enumerate(simp):
            spikes.append(spike)
            if i > 200:
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

