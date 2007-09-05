from __future__ import division
"""
spyke.gui.plot - Plotting elements
"""

__author__ = 'Reza Lotun'

import random
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib.numerix as nx

import spyke.surf
import spyke.stream

class PlotPanel(FigureCanvasWxAgg):
    """ A generic set of spyke plots. Meant to be a superclass of specific
    implementations of a plot panel (e.g. ChartPanel, EventPanel, etc.)
    """
    def __init__(self, frame, layout):
        FigureCanvasWxAgg.__init__(self, frame, -1, Figure())
        self._plot_setup = False
        
        self.pos = {}               # position of plots
        self.channels = {}          # plot y-data for each channel
        self.axes = {}              # axes for each channel
        
        self.num_channels = len(layout)
        self.set_plot_layout(layout)

        self.set_params()

    def set_params(self):
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)
        self.yrange = (-150, 150)

        self.colours = ['g'] * self.num_channels

    def set_plot_layout(self, layout):
        pass

    def init_plot(self, wave):
        for chan, sp in self.pos.iteritems():
            a = self.figure.add_axes(sp, axisbg='y', frameon=False, alpha=1.)
            a.plot(wave.ts,
                   wave.data[chan],
                   self.colours[chan],
                   antialiased=False,
                   linewidth=0.005,)
            a.set_ylim(self.yrange)
            self.axes[chan] = a
            self.channels[chan] = a.get_lines()[0]
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])

        # redraw the disply
        self.draw(True)

    def plot(self, waveforms):
        if not self._plot_setup:
            self.init_plot(waveforms)
            self._plot_setup = True
            return

        for chan in self.channels:
            self.channels[chan].set_ydata(waveforms.data[chan])
            self.axes[chan].set_ylim(self.yrange)
        self.draw(True)


class ChartPanel(PlotPanel):
    """ Chart window widget. Presents all channels layout out vertically. """

    def set_plot_layout(self):
        num = self.num_channels
        hMargin = 0.05
        vMargin = 0.05
        vSpace = 0.001

        width = 1 - 2 * hMargin
        height = (1 - 2 * vMargin - (num - 1) * vSpace) / num
        bot = vMargin

        for i in xrange(num):
            self.pos[i] = [hMargin, bot, width, height]
            bot += height + vSpace

class EventPanel(PlotPanel):
    """ Event window widget. Presents all channels layed out according
    to the passed in layout. """

    def set_params(self):
        PlotPanel.set_params(self)
        self.colours = []
        col = ['r', 'g', 'b']
        for i in xrange(54):
            self.colours.append(col[i % 3])

    def _set_plot_layout(self, layout):
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

    def set_plot_layout(self, layout):
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

class SortPanel(EventPanel):
    """ Sorting window widget. Presents all channels layed out according
    to the passed in layout. Also allows overplotting and some user
    interaction
    """
    pass

