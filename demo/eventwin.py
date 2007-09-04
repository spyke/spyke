from __future__ import division
"""
eventwin.py

A smattering of code demonstrating an event window for spyke
"""

__author__ = 'Reza Lotun'


import random
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib.numerix as nx
import numpy as np

import spyke.surf
import spyke.stream
from spyke.gui.plot import EventWindow, ChartWindow

class Demo(wx.App):
    def OnInit(self):
        self.frame = panel = EventWin(None, -1, 'Events', size=(500,600))
        #self.frame = panel = ChartWin(None, -1, 'Data', size=(500,600))
        self.frame.Show(True)
        #self.frame2.Show(True)
        return True

class EventWin(wx.Frame):
    def __init__(self, parent, id, title, **kwds):
        #self.filename = '/media/windows/Documents and Settings/Reza ' \
        #                'Lotun/Desktop/Surfdata/' \
        #                '87 - track 7c spontaneous craziness.srf'
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.plotPanel = EventWindow(self)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.plotPanel.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None

        self.timer.Start(100)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass

class ChartWin(wx.Frame):
    def __init__(self, parent, id, title, **kwds):
        #self.filename = '/media/windows/Documents and Settings/Reza ' \
        #                'Lotun/Desktop/Surfdata/' \
        #                '87 - track 7c spontaneous craziness.srf'
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.plotPanel = ChartWindow(self)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.plotPanel.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None

        self.timer.Start(100)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass
app = Demo()
app.MainLoop()
