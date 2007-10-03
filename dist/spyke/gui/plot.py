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

        # set layouts of the plot on the screen
        self.set_plot_layout(layout)
        self.set_params()

    def set_params(self):
        """ Set extra parameters. """
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)
        self.yrange = (-150, 150)
        self.colours = ['g'] * self.num_channels

    def set_plot_layout(self, layout):
        """ Override in subclasses. """
        pass

    def init_plot(self, wave):
        """ Set up axes """
        # self.pos is a map from channel -> [l, b, w, h] positions for plots
        for chan, sp in self.pos.iteritems():
            a = self.figure.add_axes(sp, axisbg='y', frameon=False, alpha=1.)

            # create an instance of a searchable line
            line = SpykeLine(wave.ts,
                             wave.data[chan],
                             linewidth=0.005,
                             color=self.colours[chan],
                             antialiased=False)

            # add line, initialize properties
            a.add_line(line)
            a.autoscale_view()
            a.set_ylim(self.yrange)
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])

            self.axes[chan] = a
            self.channels[chan] = a.get_lines()[0]
            

        # redraw the disply
        self.draw(True)

    def plot(self, waveforms):
        """ Plot waveforms """
        # check if we've set up our axes yet
        if not self._plot_setup:
            self.init_plot(waveforms)
            self._plot_setup = True
            return

        # update plots with new data
        for chan in self.channels:
            self.channels[chan].set_ydata(waveforms.data[chan])
            self.axes[chan].set_ylim(self.yrange)
        self.draw(True)


class ChartPanel(PlotPanel):
    """ Chart window widget. Presents all channels layout out vertically. """

    def set_params(self):
        PlotPanel.set_params(self)
        colgen = itertools.cycle(iter(['b', 'g', 'm', 'c', 'y', 'r', 'w']))
        self.colours = []
        for chan in xrange(self.num_channels):
            self.colours.append(colgen.next())

    def set_plot_layout(self, layout):
        """ Chartpanel plots are laid out vertically:

           (0,0)                                  (0,1)
              +-------------------------------------+
              |             ^ 
              |             | vMargin  
              |             v
              |         +----------------...
              |         |
              |<------->|        Plot 1
              | vMargin |
              |         +----------------...
              |                    ^
              |                    | vSpace
              |                    v
              |         +----------------...
              |         |
              |         |        Plot 2
              .
              |
              +

        """
        num = self.num_channels

        # XXX: some magic numbers that should be tweaked as desired
        hMargin = 0.05
        vMargin = 0.05
        vSpace = 0.001

        width = 1 - 2 * hMargin
        height = (1 - 2 * vMargin - (num - 1) * vSpace) / num
        bot = vMargin

        for chan, coords in layout.iteritems():
            self.pos[chan] = [hMargin, bot, width, height]
            bot += height + vSpace


class EventPanel(PlotPanel):
    """ Event window widget. Presents all channels layed out according
    to the passed in layout. 
    """
    def set_params(self):
        PlotPanel.set_params(self)
        self.colours = ['y'] * self.num_channels

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
              |        |  x,y (h)
              |        |       |
              |        +-------+
              |            ^
              |            | (b)
              |            v
              +------------------------------+
             (1,0)                          (1,1)
        """
        # project coordinates onto x and y axes repsectively
        xcoords = [x for x, y in layout.itervalues()]
        ycoords = [y for x, y in layout.itervalues()]

        # get limits on coordinates
        xmin, xmax = min(xcoords), max(xcoords)
        ymin, ymax = min(ycoords), max(ycoords)

        # base this on heuristics eventually XXX
        # define the width and height of the bounding boxes
        bh = 120
        bw = 75

        # Each plot chan is contained within a bounding box which
        # will necessarily overlap (necassarily because we want the plots
        # to overplot each other to maximize screen usage and to help
        # us indicate the firing of events. Bounding boxes will be centered
        # at the coordinates passed in the layout. Here we want to calculate
        # the lengths of this bounding box *relative to the layout coordinate
        # system*
        bound_xmin = xmin - bw / 2.
        bound_xmax = xmax + bw / 2.
        bound_ymin = ymin - bh / 2.
        bound_ymax = ymax + bh / 2.

        bound_width = bound_xmax - bound_xmin
        bound_height = bound_ymax - bound_ymin

        # For each coordinate, with the given bounding boxes defined above
        # center these boxes on the coordinates, and adjust to produce
        # percentages
        self.pos = {}
        for chan, coords in layout.iteritems():
            x, y = coords
            l = abs(bound_xmin - (x - bw / 2.)) / bound_width
            b = abs(bound_ymin - (y - bh / 2.)) / bound_height
            w = bw / bound_width
            h = bh / bound_height
            self.pos[chan] = [l, b, w, h]


class SortPanel(EventPanel):
    """ Sorting window widget. Presents all channels layed out according
    to the passed in layout. Also allows overplotting and some user
    interaction
    """
    def __init__(self, *args, **kwargs):
        EventPanel.__init__(self, *args, **kwargs)
        self.spikes = {}  # spike -> [SpykeLine]s
        self.x_vals = None

    def set_params(self):
        PlotPanel.set_params(self)
        self.colours = ['g'] * self.num_channels

    def add(self, spike):
        """ (Over)plot a given spike. """

        # initialize
        if len(self.spikes.keys()) == 0:

            self.init_plot(spike)

            # always plot w.r.t. these x points
            self.x_vals = spike.ts
            
            lines = []
            for num, channel in self.channels.iteritems():
                lines.append(channel)
            self.spikes[spike] = lines

        if spike not in self.spikes:
            lines = []
            for chan, axis in self.axes.iteritems():
                line = SpykeLine(self.x_vals,
                                 spike.data[chan],
                                 linewidth=0.005,
                                 color=self.colours[chan],
                                 antialiased=False)
                axis.add_line(line)
                axis.autoscale_view()
                axis.set_ylim(self.yrange)
                lines.append(line)
            self.spikes[spike] = lines
        self.draw(True)

    def remove(self, spike):
        """ Remove the selected spike from the plot display. """
        lines = self.spikes.pop(spike)
        for chan, axis in self.axes.iteritems():
            axis.lines.remove(lines[chan])
        self.draw(True)


#####----- Tests

from spyke.layout import *
import spyke.detect

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
        self.layout = eval('Polytrode' + layout_name[-3:])()
        self.curr = self.dstream.records[0].TimeStamp


class TestWindows(wx.App):
    def OnInit(self):
        op = Opener()
        self.events = panel = TestEventWin(None, -1, 'Events', op, 
                                                            size=(200,1000))
        self.chart = panel2 = TestChartWin(None, -1, 'Chart', op, 
                                                            size=(500,600))
        self.sort = panel3 = TestSortWin(None, -1, 'Data', op, 
                                                            size=(200,1000))
        self.SetTopWindow(self.events)
        self.chart.Show(True)
        self.sort.Show(True)
        return True


class PlayWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.stream = op.dstream
        self.layout = op.layout
        self.incr = 1000        # 1 ms
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)

    def onTimerEvent(self, evt):
        pass

    def onEraseBackground(self, evt):
        # prevent redraw flicker
        pass


class TestSortWin(PlayWin):
    """ Reference implementation of a test Event Window. An EventPanel
    is simply embedded in a wx Frame, and a passed in data file is "played".
    """
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)

        self.incr = 1000
        simp = spyke.detect.SimpleThreshold(self.stream, 
                                            self.stream.records[0].TimeStamp)

        self.event_iter = iter(simp)

        self.plotPanel = SortPanel(self, self.op.layout.SiteLoc)

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
        self.plotPanel.add(waveforms)


class TestEventWin(PlayWin):
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)

        self.plotPanel = EventPanel(self, self.layout.SiteLoc)
        
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = op.curr
        self.timer.Start(200)

    def onTimerEvent(self, evt):
        waveforms = self.stream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(waveforms)


class TestChartWin(PlayWin):
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)
        self.plotPanel = ChartPanel(self, self.layout.SiteLoc)
        
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = op.curr
        self.incr = 5000
        self.timer.Start(200)

    def onTimerEvent(self, evt):
        waveforms = self.stream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(waveforms)


if __name__ == '__main__':
    app = TestWindows()
    app.MainLoop()

