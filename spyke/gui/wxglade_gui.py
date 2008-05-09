#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# generated by wxGlade 0.6.3 on Fri May 09 14:45:35 2008

import wx

# begin wxGlade: extracode
wx.ID_SPIKEWIN = wx.NewId()
wx.ID_CHARTWIN = wx.NewId()
wx.ID_LFPWIN = wx.NewId()
# end wxGlade



class SpykeFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: SpykeFrame.__init__
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.notebook = wx.Notebook(self, -1, style=0)
        self.validate_pane = wx.Panel(self.notebook, -1)
        self.rip_pane = wx.Panel(self.notebook, -1)
        self.templates_pane = wx.Panel(self.notebook, -1)
        self.events_pane = wx.Panel(self.notebook, -1)
        self.events_sizer_staticbox = wx.StaticBox(self.events_pane, -1, "stuff")
        self.template_sizer_staticbox = wx.StaticBox(self.templates_pane, -1, "stuff")
        self.rip_sizer_staticbox = wx.StaticBox(self.rip_pane, -1, "stuff")
        self.validate_sizer_staticbox = wx.StaticBox(self.validate_pane, -1, "stuff")
        self.file_control_panel = wx.Panel(self, -1)
        
        # Menu Bar
        self.menubar = wx.MenuBar()
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(wx.ID_NEW, "&New\tCtrl-N", "Create new collection", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(wx.ID_OPEN, "&Open\tCtrl-O", "Open .srf or .sort file", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(wx.ID_SAVE, "&Save\tCtrl-S", "Save collection", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(wx.ID_SAVEAS, "Save As", "Save collection as", wx.ITEM_NORMAL)
        wxglade_tmp_menu.Append(wx.ID_CLOSE, "&Close\tCtrl-W", "Close .srf file", wx.ITEM_NORMAL)
        wxglade_tmp_menu.AppendSeparator()
        wxglade_tmp_menu.Append(wx.ID_EXIT, "E&xit", "Exit", wx.ITEM_NORMAL)
        self.menubar.Append(wxglade_tmp_menu, "&File")
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(wx.ID_SPIKEWIN, "Spike window", "Toggle spike window", wx.ITEM_CHECK)
        wxglade_tmp_menu.Append(wx.ID_CHARTWIN, "Chart window", "Toggle chart window", wx.ITEM_CHECK)
        wxglade_tmp_menu.Append(wx.ID_LFPWIN, "LFP window", "Toggle LFP window", wx.ITEM_CHECK)
        self.menubar.Append(wxglade_tmp_menu, "&View")
        wxglade_tmp_menu = wx.Menu()
        wxglade_tmp_menu.Append(wx.ID_ABOUT, "&About...", "Show about window", wx.ITEM_NORMAL)
        self.menubar.Append(wxglade_tmp_menu, "&Help")
        self.SetMenuBar(self.menubar)
        # Menu Bar end
        self.statusbar = self.CreateStatusBar(1, 0)
        
        # Tool Bar
        self.toolbar = wx.ToolBar(self, -1, style=wx.TB_HORIZONTAL|wx.TB_FLAT|wx.TB_TEXT)
        self.SetToolBar(self.toolbar)
        self.toolbar.AddLabelTool(wx.ID_NEW, "New", wx.Bitmap("res/new.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Create new collection", "")
        self.toolbar.AddLabelTool(wx.ID_OPEN, "Open", wx.Bitmap("res/open.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Open .srf or .sort file", "")
        self.toolbar.AddLabelTool(wx.ID_CLOSE, "Close", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Close .srf file", "Close .srf file")
        self.toolbar.AddLabelTool(wx.ID_SAVE, "Save", wx.Bitmap("res/save.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_NORMAL, "Save collection", "")
        self.toolbar.AddSeparator()
        self.toolbar.AddLabelTool(wx.ID_SPIKEWIN, "Spike", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_CHECK, "Toggle spike window", "")
        self.toolbar.AddLabelTool(wx.ID_CHARTWIN, "Chart", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_CHECK, "Toggle chart window", "")
        self.toolbar.AddLabelTool(wx.ID_LFPWIN, "LFP", wx.Bitmap("res/blank.png", wx.BITMAP_TYPE_ANY), wx.NullBitmap, wx.ITEM_CHECK, "Toggle LFP window", "")
        # Tool Bar end
        self.slider = wx.Slider(self.file_control_panel, -1, 0, 0, 10, style=wx.SL_HORIZONTAL|wx.SL_LABELS|wx.SL_SELRANGE)
        self.play_backward_button = wx.ToggleButton(self.file_control_panel, -1, "<")
        self.play_forward_button = wx.ToggleButton(self.file_control_panel, -1, ">")
        self.pause_button = wx.ToggleButton(self.file_control_panel, -1, "||")
        self.stop_button = wx.Button(self.file_control_panel, -1, ".")
        self.page_left_button = wx.Button(self.file_control_panel, -1, "|<<")
        self.step_left_button = wx.Button(self.file_control_panel, -1, "|<")
        self.step_right_button = wx.Button(self.file_control_panel, -1, ">|")
        self.page_right_button = wx.Button(self.file_control_panel, -1, ">>|")
        self.radio_box_1 = wx.RadioBox(self.events_pane, -1, "radio_box_1", choices=["choice 1", "choice 2", "choice 3"], majorDimension=0, style=wx.RA_SPECIFY_ROWS)
        self.choice_1 = wx.Choice(self.events_pane, -1, choices=[])
        self.spin_button_2 = wx.SpinButton(self.events_pane, -1 )
        self.radio_box_1_copy = wx.RadioBox(self.templates_pane, -1, "radio_box_1", choices=["choice 1", "choice 2", "choice 3"], majorDimension=0, style=wx.RA_SPECIFY_ROWS)
        self.choice_1_copy = wx.Choice(self.templates_pane, -1, choices=[])
        self.spin_button_2_copy = wx.SpinButton(self.templates_pane, -1 )

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_MENU, self.OnNew, id=wx.ID_NEW)
        self.Bind(wx.EVT_MENU, self.OnOpen, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnSaveAs, id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.OnSpike, id=wx.ID_SPIKEWIN)
        self.Bind(wx.EVT_MENU, self.OnChart, id=wx.ID_CHARTWIN)
        self.Bind(wx.EVT_MENU, self.OnLFP, id=wx.ID_LFPWIN)
        self.Bind(wx.EVT_MENU, self.OnAbout, id=wx.ID_ABOUT)
        self.Bind(wx.EVT_TOOL, self.OnNew, id=wx.ID_NEW)
        self.Bind(wx.EVT_TOOL, self.OnOpen, id=wx.ID_OPEN)
        self.Bind(wx.EVT_TOOL, self.OnClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_TOOL, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_TOOL, self.OnSpike, id=wx.ID_SPIKEWIN)
        self.Bind(wx.EVT_TOOL, self.OnChart, id=wx.ID_CHARTWIN)
        self.Bind(wx.EVT_TOOL, self.OnLFP, id=wx.ID_LFPWIN)
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: SpykeFrame.__set_properties
        self.SetTitle("spyke")
        self.SetSize((300, 300))
        self.statusbar.SetStatusWidths([-1])
        # statusbar fields
        statusbar_fields = ["statusbar"]
        for i in range(len(statusbar_fields)):
            self.statusbar.SetStatusText(statusbar_fields[i], i)
        self.toolbar.Realize()
        self.play_backward_button.SetMinSize((25, 25))
        self.play_backward_button.SetToolTipString("Toggle backward play")
        self.play_forward_button.SetMinSize((25, 25))
        self.play_forward_button.SetToolTipString("Toggle forward play")
        self.pause_button.SetMinSize((25, 25))
        self.pause_button.SetToolTipString("Toggle event detection")
        self.stop_button.SetMinSize((25, 25))
        self.stop_button.SetToolTipString("Stop")
        self.page_left_button.SetMinSize((25, 25))
        self.page_left_button.SetToolTipString("Page left")
        self.step_left_button.SetMinSize((25, 25))
        self.step_left_button.SetToolTipString("Step left")
        self.step_right_button.SetMinSize((25, 25))
        self.step_right_button.SetToolTipString("Step right")
        self.page_right_button.SetMinSize((25, 25))
        self.page_right_button.SetToolTipString("Page right")
        self.file_control_panel.Enable(False)
        self.radio_box_1.SetSelection(0)
        self.events_pane.SetToolTipString("Event detection step")
        self.radio_box_1_copy.SetSelection(0)
        self.templates_pane.SetToolTipString("Spike template generation step")
        self.templates_pane.Enable(False)
        self.rip_pane.SetToolTipString("Template ripping step")
        self.rip_pane.Enable(False)
        self.validate_pane.SetToolTipString("Validation step")
        self.validate_pane.Enable(False)
        self.notebook.Enable(False)
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: SpykeFrame.__do_layout
        spykeframe_sizer = wx.BoxSizer(wx.VERTICAL)
        validate_sizer = wx.StaticBoxSizer(self.validate_sizer_staticbox, wx.HORIZONTAL)
        rip_sizer = wx.StaticBoxSizer(self.rip_sizer_staticbox, wx.HORIZONTAL)
        template_sizer = wx.StaticBoxSizer(self.template_sizer_staticbox, wx.HORIZONTAL)
        events_sizer = wx.StaticBoxSizer(self.events_sizer_staticbox, wx.HORIZONTAL)
        file_control_sizer = wx.BoxSizer(wx.VERTICAL)
        file_control_buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        file_control_sizer.Add(self.slider, 0, wx.EXPAND, 0)
        file_control_buttons_sizer.Add(self.play_backward_button, 0, 0, 0)
        file_control_buttons_sizer.Add(self.play_forward_button, 0, 0, 0)
        file_control_buttons_sizer.Add(self.pause_button, 0, 0, 0)
        file_control_buttons_sizer.Add(self.stop_button, 0, 0, 0)
        file_control_buttons_sizer.Add((15, 25), 0, 0, 0)
        file_control_buttons_sizer.Add(self.page_left_button, 0, 0, 0)
        file_control_buttons_sizer.Add(self.step_left_button, 0, 0, 0)
        file_control_buttons_sizer.Add(self.step_right_button, 0, 0, 0)
        file_control_buttons_sizer.Add(self.page_right_button, 0, 0, 0)
        file_control_sizer.Add(file_control_buttons_sizer, 1, 0, 0)
        self.file_control_panel.SetSizer(file_control_sizer)
        spykeframe_sizer.Add(self.file_control_panel, 0, wx.EXPAND, 1)
        events_sizer.Add(self.radio_box_1, 0, 0, 0)
        events_sizer.Add(self.choice_1, 0, 0, 0)
        events_sizer.Add(self.spin_button_2, 0, 0, 0)
        self.events_pane.SetSizer(events_sizer)
        template_sizer.Add(self.radio_box_1_copy, 0, 0, 0)
        template_sizer.Add(self.choice_1_copy, 0, 0, 0)
        template_sizer.Add(self.spin_button_2_copy, 0, 0, 0)
        self.templates_pane.SetSizer(template_sizer)
        self.rip_pane.SetSizer(rip_sizer)
        self.validate_pane.SetSizer(validate_sizer)
        self.notebook.AddPage(self.events_pane, "Events")
        self.notebook.AddPage(self.templates_pane, "Templates")
        self.notebook.AddPage(self.rip_pane, "Rip")
        self.notebook.AddPage(self.validate_pane, "Validate")
        spykeframe_sizer.Add(self.notebook, 1, wx.EXPAND, 0)
        self.SetSizer(spykeframe_sizer)
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

    def OnClose(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnClose' not implemented!"
        event.Skip()

    def OnExit(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnExit' not implemented!"
        event.Skip()

    def OnSpike(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnSpike' not implemented!"
        event.Skip()

    def OnChart(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnChart' not implemented!"
        event.Skip()

    def OnLFP(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnLFP' not implemented!"
        event.Skip()

    def OnAbout(self, event): # wxGlade: SpykeFrame.<event_handler>
        print "Event handler `OnAbout' not implemented!"
        event.Skip()

# end of class SpykeFrame


class DataFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: DataFrame.__init__
        kwds["style"] = wx.CAPTION|wx.CLOSE_BOX|wx.MAXIMIZE_BOX|wx.SYSTEM_MENU|wx.RESIZE_BORDER|wx.FRAME_TOOL_WINDOW|wx.FRAME_NO_TASKBAR|wx.CLIP_CHILDREN
        wx.Frame.__init__(self, *args, **kwds)

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: DataFrame.__set_properties
        self.SetTitle("data window")
        self.SetSize((175, 675))
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: DataFrame.__do_layout
        dataframe_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(dataframe_sizer)
        self.Layout()
        # end wxGlade

# end of class DataFrame


if __name__ == "__main__":
    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    spykeframe = SpykeFrame(None, -1, "")
    app.SetTopWindow(spykeframe)
    spykeframe.Show()
    app.MainLoop()
