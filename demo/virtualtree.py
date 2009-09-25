"""Demo of how to use VirtualTree to speed up a massive tree control"""

import wx
from wx.lib.mixins.treemixin import VirtualTree


class MyTree(VirtualTree, wx.TreeCtrl):
    """Virtual tree control"""
    def OnGetItemText(self, index):
        """index is tuple of 0-based (root, child, child, ...) indices.
        An empty tuple () represents the hidden root item"""
        if len(index) == 0:
            return ''
        elif len(index) == 1:
            return str(index[0])
        else: # len(index) == 2:
            return str(index[1])

    def OnGetChildrenCount(self, index):
        """index is tuple of 0-based (root, child, child, ...) indices.
        An empty tuple () represents the hidden root item"""
        if len(index) == 0: # hidden root has how many children?
            return 10
        elif len(index) == 1:
            return 3000
        else:
            return 0


class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="simple tree", size=(400,500))
        self.tree = MyTree(self, style=wx.TR_HAS_BUTTONS|wx.TR_LINES_AT_ROOT|wx.TR_MULTIPLE|wx.TR_HIDE_ROOT|wx.TR_MULTIPLE|wx.TR_DEFAULT_STYLE|wx.NO_BORDER|wx.WANTS_CHARS)
        root = self.tree.AddRoot("wx.Object")
        self.tree.RefreshItems()

app = wx.PySimpleApp()
frame = TestFrame()
frame.Show()
app.MainLoop()
