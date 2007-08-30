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

import spyke.surf
import spyke.stream

class Demo(wx.App):
    def OnInit(self):
        self.frame = panel = EventWin(None, -1, 'Events', size=(500,600))
        #self.frame = panel = EventWin(None, -1, 'Events')
        self.frame.Show(True)
        return True

class EventWin(wx.Frame):
    def __init__(self, parent, id, title, **kwds):
        #self.filename = '/media/windows/Documents and Settings/Reza ' \
        #                'Lotun/Desktop/Surfdata/' \
        #                '87 - track 7c spontaneous craziness.srf'
        self.filename = '../data/smallSurf'
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.plotPanel = FigureCanvasWxAgg(self, -1, Figure())
        #self.plotPanel.figure.set_edgecolor('white')
        self.plotPanel.figure.set_facecolor('black')
        self.plotPanel.SetBackgroundColour(wx.BLACK)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None

        self.channels = {}
        self._initSpyke()
        self.timer.Start(100)

    def _initSpyke(self):
        self.datafile = spyke.surf.File(self.filename)
        self.datafile.parse()
        self.colours = []
        #col = ['r', 'g', 'b']
        #for i in xrange(54):
        #    self.colours.append(col[i % 3])
        self.dstream = spyke.stream.Stream(self.datafile.highpassrecords)
        self.curr = self.dstream.records[0].TimeStamp
        self.incr = 1000

        self.init_plot()

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass

    def update(self):
        self.window = self.dstream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        for chan in self.channels:
            self.channels[chan].set_ydata(self.window.data[chan])
            self.axes[chan].set_ylim((1650, 2400))
            #print self.axes[chan].get_ylim()
            #print self.axes[chan].get_autoscale_on()
        self.plotPanel.draw(True)

    def onTimerEvent(self, evt):
        self.update()

    def init_plot(self):

        ############
        #
        num = 54
        spacing = [0.00, 0.00, 0.00, 0.00]
        #offset = 0.02
        #overlap = 0.02
        offset = 0.0
        overlap = 0.0
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
        self.axes = {}
        for i in range(num // 2):
            sp=[horizMargin, bot - offset, width, height]
            a = fig.add_axes(sp, axisbg='y', frameon=False, alpha=1.)
            #a.set_ylim((0, MAXY))
            #a.plot(self.window.ts,
            #       self.window.data[chan],
            #       self.colours[chan],
            #       antialiased=False,
            #       linewidth=0.05)
            #a.set_aspect('equal')
            #a.set_adjustable('box')
            a.plot(self.window.data[chan],
                   'g',
                   antialiased=False,
                   linewidth=0.05,)
                   #scaley=False)
            #a.set_autoscale_on(False)
            self.axes[chan] = a
            self.channels[chan] = a.get_lines()[0]
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])

            # next channel
            chan += 1

            # XXX: duplicate code here - find a graceful way to code this
            # once
            sp=[horizMargin + width + hSep - overlap,
                                        bot, width, height]


            a = fig.add_axes(sp, axisbg='y', frameon=False, alpha=1.)
            a.set_adjustable('box')
            #a.set_aspect('equal')
            #a.set_ylim((0, MAXY))
            #a.plot(self.window.ts,
            #       self.window.data[chan],
            #       self.colours[chan],
            #       antialiased=False,
            #       linewidth=0.05)
            a.plot(self.window.data[chan],
                   'g',
                   antialiased=False,
                   linewidth=0.05,)
                   #scaley=False)
            #a.set_autoscale_on(False)
            self.channels[chan] = a.get_lines()[0]
            self.axes[chan] = a
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])
            bot += height + vSep

            # next channel
            chan += 1

        if self.borderAxes:
            a = fig.sca(self.borderAxes)
            a.set_frame_on(True)

        # redraw the disply
        self.plotPanel.draw(True)

app = Demo()
app.MainLoop()
