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
import spyke.detect
from spyke.probes import *
from spyke.gui.plot import EventPanel, ChartPanel

class Demo(wx.App):
    def OnInit(self):
        op = Opener()
        self.frame = panel = EventWin(None, -1, 'Events', op, size=(500,600))
        self.frame2 = panel2 = ChartWin(None, -1, 'Data', op, size=(500,600))
        self.SetTopWindow(self.frame)
        self.frame.Show(True)
        self.frame2.Show(True)
        return True

class Opener(object):
    def __init__(self):
        filename = 'C:\Documents and Settings\Reza Lotun\Desktop\Surfdata\87 - track 7c spontaneous craziness.srf'
        #filename = '/media/windows/Documents and Settings/Reza ' \
         #               'Lotun/Desktop/Surfdata/' \
         #               '87 - track 7c spontaneous craziness.srf'
        #filename = '/home/rlotun/spyke/data/smallSurf'
        #filename = '/Users/rlotun/work/spyke/data/smallSurf'
        surf_file = spyke.surf.File(filename)
        surf_file.parse()
        self.dstream = spyke.stream.Stream(surf_file.highpassrecords)
        layout_name = surf_file.layoutrecords[0].electrode_name
        layout_name = layout_name.replace('\xb5', 'u') # replace 'micro' symbol with 'u'
        self.layout = eval(layout_name)() # UNTESTED
        self.curr = self.dstream.records[0].TimeStamp

class EventWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)

        self.incr = 1000
        self.op = op
        simp = spyke.detect.SimpleThreshold(self.op.dstream, self.op.dstream.records[0].TimeStamp)
        self.event_iter = iter(simp)

        self.plotPanel = EventPanel(self, self.op.layout.SiteLoc)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None

        self.timer.Start(500)

    def onTimerEvent(self, evt):
        #waveforms = self.dstream[self.curr:self.curr+self.incr]
        #self.curr += self.incr
        waveforms = self.event_iter.next()
        print waveforms
        #print waveforms.data.shape, len(waveforms.ts)
        self.plotPanel.plot(waveforms)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass

class PlayWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        #self.filename = '/media/windows/Documents and Settings/Reza ' \
        #                'Lotun/Desktop/Surfdata/' \
        #                '87 - track 7c spontaneous craziness.srf'
        wx.Frame.__init__(self, parent, id, title, **kwds)

        self.plotPanel = EventPanel(self, self.op.layout.SiteLoc)
        self.incr = 1000
        self.op = op
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = self.op.curr
        self.timer.Start(200)

    def onTimerEvent(self, evt):
        waveforms = self.op.dstream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(waveforms)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass

class ChartWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        #self.filename = '/media/windows/Documents and Settings/Reza ' \
        #                'Lotun/Desktop/Surfdata/' \
        #                '87 - track 7c spontaneous craziness.srf'
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.op = op
        self.plotPanel = ChartPanel(self, self.op.layout.SiteLoc)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = self.op.curr
        self.incr = 5000
        self.timer.Start(200)

    def onTimerEvent(self, evt):
        waveforms = self.op.dstream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(waveforms)

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass
app = Demo()
app.MainLoop()
