#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# generated by wxGlade 0.6.3 on Sat Apr 26 16:28:18 2008

import wx

# begin wxGlade: extracode
# end wxGlade



class SpykeFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: SpykeFrame.__init__
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.notebook = wx.Notebook(self, -1, style=0)
        
        # Menu Bar
        self.menubar = wx.MenuBar()
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(1, "&New", "Create new collection", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(2, "&Open", "Open .srf or .sort file", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(3, "&Save", "Save collection", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(4, "Save As", "Save collection as", wx.ITEM_NORMAL)
        wxglade_tmp_menu.AppendSeparator()
        wxglade_tmp_menu.Append(5, "E&xit", "Exit", wx.ITEM_NORMAL)
        self.menubar.Append(wxglade_tmp_menu, "&File")
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(10, "Chart window", "Toggle chart window", wx.ITEM_CHECK)
        wxglade_tmp_menu.Append(11, "Spike window", "Toggle spike window", wx.ITEM_CHECK)
        wxglade_tmp_menu.Append(12, "EEG window", "Toggle EEG window", wx.ITEM_CHECK)
        self.menubar.Append(wxglade_tmp_menu, "&View")
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(6, "&About...", "Show about window", wx.ITEM_NORMAL)
        self.menubar.Append(wxglade_tmp_menu, "&Help")
        self.SetMenuBar(self.menubar)
        # Menu Bar end
        self.statusbar = self.CreateStatusBar(1, 0)
        
        # Tool Bar
        self.toolbar = wx.ToolBar(self, -1, style=wx.TB_HORIZONTAL|wx.TB_FLAT|wx.TB_TEXT|wx.TB_HORZ_LAYOUT|wx.TB_HORZ_TEXT)
        self.SetToolBar(self.toolbar)
        self.toolbar.AddLabelTool(1, "New", wx.Bitmap("res/new.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Create new collection", "")
        self.toolbar.AddLabelTool(2, "Open", wx.Bitmap("res/open.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Open .srf or .sort file", "")
        self.toolbar.AddLabelTool(3, "Save", wx.Bitmap("res/save.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Save collection", "")
        self.toolbar.AddSeparator()
        self.toolbar.AddLabelTool(10, "Chart", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_CHECK, "Toggle chart window", "")
        self.toolbar.AddLabelTool(11, "Spike", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_CHECK, "Toggle spike window", "")
        self.toolbar.AddLabelTool(12, "EEG", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_CHECK, "Toggle EEG window", "")
        # Tool Bar end
        self.events_pane = wx.Panel(self.notebook, -1)
        self.templates_pane = wx.Panel(self.notebook, -1)
        self.rip_pane = wx.Panel(self.notebook, -1)
        self.validate_pane = wx.Panel(self.notebook, -1)

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_MENU, self.OnNew, id=1)
        self.Bind(wx.EVT_MENU, self.OnOpen, id=2)
        self.Bind(wx.EVT_MENU, self.OnSave, id=3)
        self.Bind(wx.EVT_MENU, self.OnSaveAs, id=4)
        self.Bind(wx.EVT_MENU, self.OnExit, id=5)
        self.Bind(wx.EVT_MENU, self.OnChart, id=10)
        self.Bind(wx.EVT_MENU, self.OnSpike, id=11)
        self.Bind(wx.EVT_MENU, self.OnEEG, id=12)
        self.Bind(wx.EVT_MENU, self.OnAbout, id=6)
        self.Bind(wx.EVT_TOOL, self.OnNew, id=1)
        self.Bind(wx.EVT_TOOL, self.OnOpen, id=2)
        self.Bind(wx.EVT_TOOL, self.OnSave, id=3)
        self.Bind(wx.EVT_TOOL, self.OnChart, id=10)
        self.Bind(wx.EVT_TOOL, self.OnSpike, id=11)
        self.Bind(wx.EVT_TOOL, self.OnEEG, id=12)
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: SpykeFrame.__set_properties
        self.SetTitle("spyke")
        self.SetSize((459, 392))
        self.statusbar.SetStatusWidths([-1])
        # statusbar fields
        statusbar_fields = ["statusbar"]
        for i in range(len(statusbar_fields)):
            self.statusbar.SetStatusText(statusbar_fields[i], i)
        self.toolbar.Realize()
        self.events_pane.SetToolTipString("Event detection step")
        self.templates_pane.SetToolTipString("Spike template generation step")
        self.rip_pane.SetToolTipString("Template ripping step")
        self.validate_pane.SetToolTipString("Validation step")
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: SpykeFrame.__do_layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.notebook.AddPage(self.events_pane, "Events")
        self.notebook.AddPage(self.templates_pane, "Templates")
        self.notebook.AddPage(self.rip_pane, "Rip")
        self.notebook.AddPage(self.validate_pane, "Validate")
        sizer.Add(self.notebook, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        self.Layout()
        # end wxGlade

    def OnNew(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnNew' not implemented!"
        event.Skip()

    def OnOpen(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnOpen' not implemented!"
        event.Skip()

    def OnSave(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnSave' not implemented!"
        event.Skip()

    def OnSaveAs(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnSaveAs' not implemented!"
        event.Skip()

    def OnExit(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnExit' not implemented!"
        event.Skip()

    def OnChart(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnChart' not implemented!"
        event.Skip()

    def OnSpike(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnSpike' not implemented!"
        event.Skip()

    def OnEEG(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnEEG' not implemented!"
        event.Skip()

    def OnAbout(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnAbout' not implemented!"
        event.Skip()

# end of class SpykeFrame


if __name__ == "__main__":
    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    spykeframe = SpykeFrame(None, -1, "")
    app.SetTopWindow(spykeframe)
    spykeframe.Show()
    app.MainLoop()
