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
import matplotlib.patches

import spyke.surf
import spyke.stream

class Demo(wx.App):
    def OnInit(self):
        self.frame = panel = EventWin(None, -1, 'Events', size=(800,600))
        self.frame.Show(True)
        return True

class EventWin(wx.Frame):
    def __init__(self, parent, id, title, **kwds):
        self.filename = '/media/windows/Documents and Settings/Reza ' \
                        'Lotun/Desktop/Surfdata/' \
                        '87 - track 7c spontaneous craziness.srf'

        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.plotPanel = FigureCanvasWxAgg(self, -1, Figure((16.0, 13.70), 96))
        self.plotPanel.figure.set_edgecolor('white')
        self.plotPanel.figure.set_facecolor('black')
        self.plotPanel.SetBackgroundColour(wx.BLACK)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        #self.plotPanel = wxmpl.PlotPanel(self, -1)
        self.borderAxes = None

        #wxmpl.EVT_SELECTION(self, self.plotPanel.GetId(), self._on_selection)
        #wxmpl.EVT_POINT(self, self.plotPanel.GetId(), self._on_point)

        self._initSpyke()
        #self._layout()
        self._replot()
        self.timer.Start(100)

    def _initSpyke(self):
        self.datafile = spyke.surf.File(self.filename)
        self.datafile.parse()
        self.colours = []
        col = ['r', 'g', 'b']
        for i in xrange(54):
            self.colours.append(col[i % 3])
        self.dstream = spyke.stream.Stream(self.datafile.highpassrecords)
        self.curr = self.dstream.records[0].TimeStamp
        self.incr = 1000

    def onTimerEvent(self, evt):
        self._replot()

    def _layout(self):
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer.Add((1, 1), 1, 0, 0)
        btnSizer.Add(self.regionButton, 0, wx.BOTTOM|wx.RIGHT, 5)
        btnSizer.Add(self.pointButton,  0, wx.BOTTOM|wx.RIGHT, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.plotPanel, 0, wx.ALL, 5)
        sizer.Add(btnSizer, 0, wx.EXPAND, 0)

        self.SetSizer(sizer)
        self.Fit()

    def _replot(self):
        _gr = {}
        _ax = 0
        def _drawGraph(chan, col, sp, axisbg, frameon):
            global _ax
            a = fig.add_axes(sp,
                             axisbg,
                             frameon=False)
            a.clear()
            #_gr[_ax] = a
            #_ax += 1
            a.plot(self.t, self.v, col, antialiased=False, linewidth=0.05)
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])

        ############
        #
        num = 54
        spacing = [0.3, 0.05, 0.01, 0.01]
        offset = 0.02
        overlap = 0.02
        #
        #############


        fig = self.plotPanel.figure
        #fig.clear()
        horizMargin, vertMargin, hSep, vSep = spacing
        width = (1 - 2 * horizMargin - hSep) / 2
        n = num / 2
        height = (1 - 2 * vertMargin - (n - 1) * vSep) / n
        bot = vertMargin
        chan = 0

        self.window = self.dstream[self.curr:self.curr+self.incr]
        self.curr += self.incr

        for i in range(num // 2):
            self.v = self.window.data[chan]
            self.t = self.window.ts
            _drawGraph(chan, self.colours[chan], sp=[horizMargin, bot - offset, width, height],
                     axisbg='y',
                     frameon=False)

            chan += 1

            self.v = self.window.data[chan]
            _drawGraph(chan, self.colours[chan], sp=[horizMargin + width + hSep - overlap,
                                        bot, width, height],
                      axisbg='y', frameon=False)
            bot += height + vSep

        if self.borderAxes:
            a = fig.sca(self.borderAxes)
            a.set_frame_on(True)

        # redraw the disply
        self.plotPanel.draw(True)

    def _on_regionButton(self, evt):
        if self.regionButton.GetValue():
            self.plotPanel.set_zoom(False)
        else:
            self.plotPanel.set_zoom(True)

    def _on_selection(self, evt):
        self.plotPanel.set_zoom(True)
        #self.regionButton.SetValue(False)

        x1, y1 = evt.x1data, evt.y1data
        x2, y2 = evt.x2data, evt.y2data

        self.selectionPoints.append(((x1, y1), x2-x1, y2-y1))
        self._replot()

    def _on_point(self, evt):
        self.borderAxes = evt.axes
        self._replot()

app = Demo()
app.MainLoop()
