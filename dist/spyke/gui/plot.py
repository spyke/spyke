from __future__ import division
"""
spyke.gui.plot - Plotting elements
"""

__author__ = 'Reza Lotun'

import itertools
import random

import wx

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.numerix as nx


import spyke.surf
import spyke.stream

class SpykeLine(Line2D):
    """ Line2D's that can be compared to each other for equality. """

    def __hash__(self):
        """ Hash the string representation of the y data. """
        return hash(str(self._y))

    def __eq__(self, other):
        return hash(self) == hash(other)


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
            line = SpykeLine(wave.ts,
                             wave.data[chan],
                             linewidth=0.005,
                             color=self.colours[chan],
                             antialiased=False)
            a.cla()
            a.add_line(line)
            a.autoscale_view()
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
            #self.channels[chan].draw()
        self.draw(True)


class ChartPanel(PlotPanel):
    """ Chart window widget. Presents all channels layout out vertically. """

    def set_plot_layout(self, layout):
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

    def set_plot_layout(self, layout):
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

        # base this on heuristics eventually
        bh = 100
        bw = 75

        bound_xmin = xmin - bw / 2.
        bound_xmax = xmax + bw / 2.
        bound_ymin = ymin - bh / 2.
        bound_ymax = ymax + bh / 2.

        bound_width = bound_xmax - bound_xmin
        bound_height = bound_ymax - bound_ymin

        self.pos = {}
        for chan, coords in layout.iteritems():
            x, y = coords
            l = abs(bound_xmin - (x - bw / 2.)) / bound_width
            b = abs(bound_ymin - (y - bh / 2.)) / bound_height
            w = bw / bound_width
            h = bh / bound_height
            self.pos[chan] = [l, b, w, h]
            print chan, [l, b, w, h]


class SortPanel(EventPanel):
    """ Sorting window widget. Presents all channels layed out according
    to the passed in layout. Also allows overplotting and some user
    interaction
    """
    def __init__(self, *args, **kwargs):
        EventPanel.__init__(self, *args, **kwargs)
        self.cycle_colours = itertools.cycle(iter(['b', 'g', 'm', 'c', 'y', 'r', 'w']))
        self.spikes = {}  # spike -> [SpykeLine]s
        self.x_vals = None

    def add(self, spike):
        if len(self.spikes.keys()) == 0:
            # initialize our plot
            self.init_plot(spike)
            self.x_vals = spike.ts
            lines = []
            for num, channel in self.channels.iteritems():
                lines.append(channel)
            self.spikes[spike] = lines

        if spike not in self.spikes:
            lines = []
            colour = self.cycle_colours.next()
            for chan, axis in self.axes.iteritems():
                line = SpykeLine(self.x_vals,
                                 spike.data[chan],
                                 linewidth=0.005,
                                 color=colour,
                                 antialiased=False)
                axis.add_line(line)
                axis.autoscale_view()
                axis.set_ylim(self.yrange)
                lines.append(line)
            self.spikes[spike] = lines
        self.draw(True)

    def remove(self, spike):
        lines = self.spikes.pop(spike)
        for chan, axis in self.axes.iteritems():
            axis.lines.remove(lines[chan])
        self.draw(True)


