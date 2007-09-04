from __future__ import division
"""
spyke.plot - Plotting elements
"""

__author__ = 'Reza Lotun'

import random
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib.numerix as nx

import spyke.surf
import spyke.stream

def calc_spacings(layout, figh, figw):
    """ Map from polytrode locations given as (x, y) coordinates
    into position information for the spike plots, which are stored
    as a list of four values [l, b, w, h]. To illustrate this, consider
    loc_i = (x, y) are the coordinates for the polytrode on channel i.
    We want to map these coordinates to the unit square.
       (0,0)                          (0,1)
          +------------------------------+
          |        +--(w)--+
          |<-(l)-> |       |
          |        | loc_i (h)
          |        |       |
          |        +-------+
          |            ^
          |            | (b)
          |            v
          +------------------------------+
         (1,0)                          (1,1)
    """

    xcoords = [x for x, y in layout.itervalues()]
    ycoords = [y for x, y in layout.itervalues()]
    xmin, xmax = min(xcoords), max(xcoords)
    ymin, ymax = min(ycoords), max(ycoords)
    boxwidth, boxheight = xmax - xmin, ymax - ymin
    numchannels = len(xcoords)
    columns = len(set(xcoords))
    num_per_column = numchannels / columns

    ########
    rl_margin, tb_margin, hsep, vsep = 0.01, 0.01, 0.01, 0.01
    w = (1 - 2 * rl_margin - hsep) / columns
    h = (1 - 2 * tb_margin - (num_per_column - 1) * vsep) / num_per_column
    ########

    plotwidth = w * boxwidth
    plotheight = h * boxheight
    rl_margin_l = rl_margin * box_width
    tb_margin_l = tb_margin * box_height
    hsep_l = hsep * boxwidth
    vsep_l = vsep * boxheigth

    new_box_width = plotwidth * columns + hsep_l + 2 * rl_margin_l
    new_box_height = plotheight * num_per_column  \
                        * (num_per_column -1) * vsep_l \
                        * tb_margin_l * 2

    pos = {}
    for chan, loc in layout.iteritems():
        x, y = loc

        # XXX: make this better
        if x == xmin:
            col = 1
        elif x == xmax:
            col = numcols
        elif x < xmax:
            col = numcols - 1

        #x = x - xmin + rl_margin_l + plotwidth / 2
        #y = y - ymin + tb_margin_l +

class PlotPanel(FigureCanvasWxAgg):
    """ A generic set of spyke plots. Meant to be a superclass of specific
    implementations of a plot panel (e.g. ChartPanel, EventPanel, etc.)
    """
    def __init__(self, frame):
        self.filename = '../data/smallSurf'
        FigureCanvasWxAgg.__init__(self, frame, -1, Figure())

        self.num_channels = 54 # XXX: hack for now
        self.pos = {}
        self.channels = {}
        self.dstream = {}

        self._initSpyke()

        self.set_colours()
        self.set_channel_layout()
        self.init_plot()


    def set_colours(self):
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)

    def set_channel_layout(self):
        pass

    def redrawPlots(self):
        for chan in self.channels:
            self.figure.draw_artist(self.axes[chan])
        #self.draw(True)

    def update(self):
        self.window = self.dstream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        for chan in self.channels:
            #data = self.window.data[chan] - 2048+500*chan / 500.
            #self.channels[chan].set_ydata(data)
            self.channels[chan].set_ydata(self.window.data[chan])
            self.axes[chan].set_ylim((-150, 150))
            #print self.axes[chan].get_ylim()
            #print self.axes[chan].get_autoscale_on()
        self.draw(True)
        #self.redrawPlots()

    def onTimerEvent(self, evt):
        self.update()

    def _initSpyke(self):
        self.datafile = spyke.surf.File(self.filename)
        self.datafile.parse()
        self.colours = []
        col = ['g', 'g', 'g']  # XXX: clean this up!
        for i in xrange(54):
            self.colours.append(col[i % 3])
        self.dstream = spyke.stream.Stream(self.datafile.highpassrecords)
        self.curr = self.dstream.records[0].TimeStamp
        self.incr = 1000

    def init_plot(self):
        self.window = self.dstream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.axes = {}

        for chan, sp in self.pos.iteritems():
            a = self.figure.add_axes(sp, axisbg='y', frameon=False, alpha=1.)
            #a.set_ylim((0, MAXY))
            #a.plot(self.window.ts,
            #       self.window.data[chan],
            #       self.colours[chan],
            #       antialiased=False,
            #       linewidth=0.05)
            #a.set_aspect('equal')
            #a.set_adjustable('box')
            a.plot(self.window.ts,
                   self.window.data[chan],
                   self.colours[chan],
                   antialiased=False,
                   linewidth=0.05,)
                   #scaley=False)
            #a.set_autoscale_on(False)
            self.axes[chan] = a
            self.channels[chan] = a.get_lines()[0]
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])

        #if self.borderAxes:
        #    a = self.figure.sca(self.borderAxes)
        #    a.set_frame_on(True)

        # redraw the disply
        self.draw(True)

class ChartWindow(PlotPanel):
    """ Chart window widget. Presents all channels layout out vertically. """

    def set_channel_layout(self):
        num = 54 # XXX: clean up!
        hMargin = 0.05
        vMargin = 0.05
        vSpace = 0.001

        width = 1 - 2 * hMargin
        height = (1 - 2 * vMargin - (num - 1) * vSpace) / num
        bot = vMargin

        for i in xrange(num):
            self.pos[i] = [hMargin, bot, width, height]
            bot += height + vSpace

    def _set_channel_layout(self):
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

        horizMargin, vertMargin, hSep, vSep = spacing
        width = (1 - 2 * horizMargin - hSep) / 2
        n = num / 2
        height = (1 - 2 * vertMargin - (n - 1) * vSep) / n
        bot = vertMargin
        chan = 0

        for i in range(num // 2):
            self.pos[chan]=[horizMargin, bot - offset, width, height]
            # next channel
            chan += 1

            self.pos[chan]=[horizMargin + width + hSep - overlap,
                                        bot, width, height]

            bot += height + vSep
            # next channel
            chan += 1

class EventWindow(PlotPanel):
    """ Event window widget. Presents all channels layed out according
    to the passed in layout. """

    def set_colours(self):
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)
        self.colours = []
        col = ['r', 'g', 'b']
        for i in xrange(54):
            self.colours.append(col[i % 3])

    def set_channel_layout(self):
        # XXX: working on mapping actually layout we'll play with this
        # hack for now
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
            self.pos[chan]=[horizMargin, bot - offset, width, height + 0.07]
            # next channel
            chan += 1

            self.pos[chan]=[horizMargin + width + hSep - overlap,
                                        bot, width, height + 0.07]

            bot += height + vSep
            # next channel
            chan += 1

class SortingWindow(EventWindow):
    """ Sorting window widget. Presents all channels layed out according
    to the passed in layout. Also allows overplotting and some user
    interaction
    """
    pass

