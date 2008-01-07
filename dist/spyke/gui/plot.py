from __future__ import division
"""
spyke.gui.plot - Plotting elements
"""

__author__ = 'Reza Lotun'

import itertools
import random

import numpy

import wx

from matplotlib import rcParams
rcParams['lines.linestyle'] = '-'
rcParams['lines.marker'] = ''

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.numerix as nx

import spyke.surf
import spyke.stream
from spyke.gui.events import *


class AxesWrapper(object):
    """ A wrapper around an axes that delegates access to attributes to an
    actual axes object. Really meant to make axes hashable.
    """
    def __init__(self, axes):
        self._spyke_axes = axes

    def __getattr__(self, name):
        # delegate access of attribs to wrapped axes
        return getattr(self._spyke_axes, name)

    def __hash__(self):
        # base the hash function on the position rect
        return hash(str(self.get_position(original=True)))

    def __eq__(self, other):
        return hash(self) == hash(other)


class SpykeLine(Line2D):
    """ Line2D's that can be compared to each other for equality. """
    def __init__(self, *args, **kwargs):
        Line2D.__init__(self, *args, **kwargs)
        self.colour = 'none'

    def __hash__(self):
        """ Hash the string representation of the y data. """
        return hash(self.colour + str(self._y))

    def __eq__(self, other):
        return hash(self) == hash(other)


class OneAxisPlotPanel(FigureCanvasWxAgg):
    """ One axis plot panel. """
    def __init__(self, frame, layout):
        FigureCanvasWxAgg.__init__(self, frame, -1, Figure())
        self._plot_setup = False
        self.layout = layout


        self.pos = {}               # position of lines
        self.channels = {}          # plot y-data for each channel
        self.axes = {}              # axes for each channel, chan -> axes
        self.axesToChan = {}        # axes -> chan

        self.num_channels = len(layout)

        # set layouts of the plot on the screen
        #self.set_plot_layout(layout)
        self.set_params()
        self.my_ylim = None
        self.my_xlim = None

    def set_plot_layout(self, layout):
        """ Override in subclasses. """
        pass

    def set_params(self):
        """ Set extra parameters. """
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)
        self.colours = ['g'] * self.num_channels

    def init_plot(self, wave, colour='g'):
        # create our one axis to rule them all
        pos = [0, 0, 1, 1]
        self.my_ax = self.figure.add_axes(pos,
                                       axisbg='b',
                                       frameon=False,
                                       alpha=1.)
        self.static_x_vals = None
        self.set_plot_layout(wave)
        self.my_ax._visible = False
        #self.ax.autoscale_view()
        #self.ax.grid(True)
        self.my_ax.set_xticks([])
        self.my_ax.set_yticks([])
        self.static_x_vals = numpy.asarray(wave.ts - numpy.asarray([min(wave.ts)] * len(wave.ts)))
        self.rza_lines = {}
        self.my_ax._autoscaleon = False
        for chan, sp in self.pos.iteritems():
            self.axes[chan] = self.my_ax
            x_off, y_off = self.pos[chan]
            line = SpykeLine(self.static_x_vals + x_off,
                             wave.data[chan] + y_off,
                             linewidth=0.005,
                             color=self.colours[chan],
                             antialiased=False)
            line.colour = self.colours[chan]
            self.rza_lines[chan] = line
            self.my_ax.add_line(line)

            self.channels[chan] = line

            # maintain reverse mapping of axes -> channels
            self.axesToChan[AxesWrapper(self.my_ax)] = chan

        self.my_ax._visible = True
        # redraw the disply
        self.draw(True)

    def plot(self, waveforms):
        """ Plot waveforms """
        # check if we've set up our axes yet
        if not self._plot_setup:
            self.init_plot(waveforms)
            self._plot_setup = True

        # update plots with new data
        for chan in self.rza_lines:
            #self.rza_lines[chan]._visible = False
            x_off, y_off = self.pos[chan]
            #line = SpykeLine(self.static_x_vals + x_off,
            #                 waveforms.data[chan] + y_off,
            #                 linewidth=0.005,
            #                 color=self.colours[chan],
            #                 antialiased=False)
            #self.ax.add_line(line)
            self.rza_lines[chan].set_ydata(waveforms.data[chan] + y_off)
            #self.rza_lines[chan].set_xdata(self.static_x_vals + x_off)
            #self.rza_lines[chan].set_data(self.static_x_vals + x_off, waveforms.data[chan] + y_off)
            #self.my_ax.set_ylim(self.my_ylim)
            #self.my_ax.set_xlim(self.my_xlim)
            self.rza_lines[chan]._visible = True
        print 'Y: ', self.my_ax.get_ylim()
        print 'X: ', self.my_ax.get_xlim()
        #self.my_ax.set_ylim(self.my_ylim)
        #self.my_ax.set_xlim(0, 1000)
        self.my_ax._visible = True
            #self.channels[chan]
            #line._visible = True
        self.draw(True)


class PlotPanel(FigureCanvasWxAgg):
    """ A generic set of spyke plots. Meant to be a superclass of specific
    implementations of a plot panel (e.g. ChartPanel, EventPanel, etc.)
    """
    def __init__(self, frame, layout):
        FigureCanvasWxAgg.__init__(self, frame, -1, Figure())
        self._plot_setup = False

        self.pos = {}               # position of plots
        self.channels = {}          # plot y-data for each channel
        self.axes = {}              # axes for each channel, chan -> axes
        self.axesToChan = {}        # axes -> chan

        self.num_channels = len(layout)

        # set layouts of the plot on the screen
        self.set_plot_layout(layout)
        self.set_params()

    def set_params(self):
        """ Set extra parameters. """
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)
        self.yrange = (-260, 260)
        self.colours = ['g'] * self.num_channels

    def set_plot_layout(self, layout):
        """ Override in subclasses. """
        pass

    def init_plot(self, wave, colour='g'):
        """ Set up axes """
        # self.pos is a map from channel -> [l, b, w, h] positions for plots
        for chan, sp in self.pos.iteritems():
            a = self.figure.add_axes(sp, axisbg='b', frameon=False, alpha=1.)

            colours = [colour] * self.num_channels
            # create an instance of a searchable line
            line = SpykeLine(wave.ts,
                             wave.data[chan],
                             linewidth=0.005,
                             color=colours[chan],
                             antialiased=False)
            line.colour = colour

            # add line, initialize properties
            a._visible = False
            a.add_line(line)
            a.autoscale_view()
            a.set_ylim(self.yrange)
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])
            a._visible = True

            self.axes[chan] = a
            self.channels[chan] = a.get_lines()[0]

            # maintain reverse mapping of axes -> channels
            self.axesToChan[AxesWrapper(a)] = chan

        # redraw the disply
        self.draw(True)

    def plot(self, waveforms):
        """ Plot waveforms """
        # check if we've set up our axes yet
        if not self._plot_setup:
            self.init_plot(waveforms)
            self._plot_setup = True

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

           (0,1)                                  (1,1)
              +-------------------------------------+
              |             ^
              |             | vMargin
              |             v
              |         +----------------...
              |         |
              |<------->|        center 1             =
              | vMargin |                              |
              |         +----------------...           | alpha
              |         |                              |
              |         +---- ..............overlap    |
              |         |        center 2             =
              .
              |
              +
           (0,0)
        """
        num = self.num_channels

        # project coordinates onto x and y axes repsectively
        xcoords = [x for x, y in layout.itervalues()]
        ycoords = [y for x, y in layout.itervalues()]

        # get limits on coordinates
        xmin, xmax = min(xcoords), max(xcoords)
        ymin, ymax = min(ycoords), max(ycoords)

        # XXX: some magic numbers that should be tweaked as desired
        hMargin = 0.05
        vMargin = 0.03

        box_height = 0.1

        # total amout of vertical buffer space (that is, vertical margins)
        vBuf = 2 * vMargin + box_height / 2 # XXX - heuristic/hack
        alpha = (1 - vBuf) / (num - 1)      # distance between centers
        width = 1 - 2 * hMargin

        # the first channel starts at the top
        center = 1 - vMargin - box_height / 4
        for chan, coords in layout.iteritems():
            bot = center - box_height / 2
            self.pos[chan] = [hMargin, bot, width, box_height]
            center -= alpha


class OneAxisChartPanel(OneAxisPlotPanel):
    def set_params(self):
        OneAxisPlotPanel.set_params(self)
        colgen = itertools.cycle(iter(['b', 'g', 'm', 'c', 'y', 'r', 'w']))
        self.colours = []
        for chan in xrange(self.num_channels):
            self.colours.append(colgen.next())

    def set_plot_layout(self, wave):
        num = self.num_channels
        # the first channel starts at the top
        self.my_ax.set_ylim(-50, 54*100 - 50)
        self.my_ax.set_xlim(min(wave.ts), max(wave.ts))
        for chan, coords in self.layout.iteritems():
            self.pos[chan] = (0, chan * 100)


class OneAxisEventPanel(OneAxisPlotPanel):
    """ Event window widget. Presents all channels layed out according
    to the passed in layout.
    """
    def set_params(self):
        OneAxisPlotPanel.set_params(self)
        self.colours = ['m'] * self.num_channels


    def set_plot_layout(self, wave):
        """ Map from polytrode locations given as (x, y) coordinates
        into position information for the spike plots, which are stored
        as a list of four values [l, b, w, h]. To illustrate this, consider
        loc_i = (x, y) are the coordinates for the polytrode on channel i.
        We want to map these coordinates to the unit square.
           (0,1)                          (1,1)
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
             (0,0)                          (0,1)
        """
        layout = self.layout
        # project coordinates onto x and y axes repsectively
        xcoords = [x for x, y in layout.itervalues()]
        ycoords = [y for x, y in layout.itervalues()]

        # get limits on coordinates
        xmin, xmax = min(xcoords), max(xcoords)
        ymin, ymax = min(ycoords), max(ycoords)

        # base this on heuristics eventually XXX
        # define the width and height of the bounding boxes
        col_width = max(wave.ts) - min(wave.ts) - 100    # slight overlap

        x_cols = list(set(xcoords))
        num_cols = len(x_cols)

        #        x           x           x
        #  -------------- ----------  -----------
        # each x should be the center of the columns
        # each columb should be min(wave.ts) - max(wave.ts)
        self.my_xlim = (min(wave.ts), num_cols*col_width)
        shifted = wave.ts - numpy.asarray([min(wave.ts)] * len(wave.ts))
        self.my_xlim = (min(shifted), num_cols*col_width)
        self.my_ax.set_xlim(self.my_xlim)


        x_offsets = {}
        for i, x in enumerate(sorted(x_cols)):
            x_offsets[x] = i * col_width
        # For each coordinate, with the given bounding boxes defined above
        # center these boxes on the coordinates, and adjust to produce
        # percentages
        y_rows = list(set(ycoords))
        num_rows = len(y_rows)
        row_height = 100
        y_offsets = {}
        self.my_ax.set_ylim(-100, num_rows*row_height)
        self.my_ylim = (-100, num_rows*row_height)
        for i, y in enumerate(sorted(y_rows)):
            y_offsets[y] = i * row_height

        self.pos = {}
        for chan, coords in layout.iteritems():
            x, y = coords

            x_off = x_offsets[x]
            y_off = y_offsets[y]
            self.pos[chan] = (x_off, y_off)

        print 'LIMITS: ', self.my_xlim, self.my_ylim

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
           (0,1)                          (1,1)
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
             (0,0)                          (0,1)
        """
        # project coordinates onto x and y axes repsectively
        xcoords = [x for x, y in layout.itervalues()]
        ycoords = [y for x, y in layout.itervalues()]

        # get limits on coordinates
        xmin, xmax = min(xcoords), max(xcoords)
        ymin, ymax = min(ycoords), max(ycoords)

        # base this on heuristics eventually XXX
        # define the width and height of the bounding boxes
        bh = 130
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


class OneAxisSortPanel(OneAxisEventPanel):
    """ Sorting window widget. Presents all channels layed out according
    to the passed in layout. Also allows overplotting and some user
    interaction
    """
    def __init__(self, *args, **kwargs):
        OneAxisEventPanel.__init__(self, *args, **kwargs)
        self.spikes = {}  # (spike, colour) -> [[SpykeLine], visible]
        #self.x_vals = None
        self._initialized = False
        self.top = 10

    def set_params(self):
        OneAxisPlotPanel.set_params(self)
        self.colours = ['r'] * self.num_channels

    def _toggleVisible(self, spike, colour, top=None):
        lines, curr_visible = self.spikes[(spike, colour)]
        curr_visible = not curr_visible

        for line in lines:
            line._visible = curr_visible
            line.zorder = self.top

        self.spikes[(spike, colour)][1] = curr_visible

    def _toggleChannels(self, spike, colour, channels):
        lines, curr_visible = self.spikes[(spike, colour)]
        for line, isVisible in zip(lines, channels):
            line._visible = isVisible

    def add(self, spike, colour, top=False, channels=None):
        """ (Over)plot a given spike. """
        colours = [colour] * self.num_channels

        if not channels:
            channels = self.num_channels * [True]

        if top:
            self.top += 0.1
        # initialize
        if not self._initialized:

            self.init_plot(spike, colour)
            print 'Done Init!'
            # always plot w.r.t. these x points
            #self.x_vals = spike.ts

            lines = []
            for num, channel in self.channels.iteritems():
                #x_off, y_off = self.pos[channel]
                lines.append(channel)
            self.spikes[(spike, colour)] = [lines, True]
            self._initialized = True

        elif (spike, colour) in self.spikes:
            self._toggleVisible(spike, colour, top)
            self._toggleChannels(spike, colour, channels)

        elif (spike, colour) not in self.spikes:
            lines = []
            for chan in self.channels:
                x_off, y_off = self.pos[chan]
                line = SpykeLine(self.static_x_vals + x_off,
                                 spike.data[chan] + y_off,
                                 linewidth=0.005,
                                 color=colours[chan],
                                 antialiased=False)
                line._visible = False
                line.colour = colour
                self.my_ax.add_line(line)
                #axis.autoscale_view()
                #axis.set_ylim(self.yrange)
                line._visible = True
                lines.append(line)
            self.spikes[(spike, colour)] = [lines, True]

        self.draw(True)

    def remove(self, spike, colour):
        """ Remove the selected spike from the plot display. """
        self._toggleVisible(spike, colour)
        self.draw(True)

SortPanel = OneAxisSortPanel

class ManyAxisSortPanel(EventPanel):
    """ Sorting window widget. Presents all channels layed out according
    to the passed in layout. Also allows overplotting and some user
    interaction
    """
    def __init__(self, *args, **kwargs):
        EventPanel.__init__(self, *args, **kwargs)
        self.spikes = {}  # (spike, colour) -> [[SpykeLine], visible]
        self.x_vals = None
        self._initialized = False
        self.top = 10

    def set_params(self):
        PlotPanel.set_params(self)
        self.colours = ['g'] * self.num_channels

    def _toggleVisible(self, spike, colour, top=None):
        lines, curr_visible = self.spikes[(spike, colour)]
        curr_visible = not curr_visible

        for line in lines:
            line._visible = curr_visible
            line.zorder = self.top

        self.spikes[(spike, colour)][1] = curr_visible

    def _toggleChannels(self, spike, colour, channels):
        lines, curr_visible = self.spikes[(spike, colour)]
        for line, isVisible in zip(lines, channels):
            line._visible = isVisible

    def add(self, spike, colour, top=False, channels=None):
        """ (Over)plot a given spike. """
        colours = [colour] * self.num_channels

        if not channels:
            channels = self.num_channels * [True]

        if top:
            self.top += 0.1
        # initialize
        if not self._initialized:

            self.init_plot(spike, colour)

            # always plot w.r.t. these x points
            self.x_vals = spike.ts

            lines = []
            for num, channel in self.channels.iteritems():
                lines.append(channel)
            self.spikes[(spike, colour)] = [lines, True]
            self._initialized = True

        elif (spike, colour) in self.spikes:
            self._toggleVisible(spike, colour, top)
            self._toggleChannels(spike, colour, channels)

        elif (spike, colour) not in self.spikes:
            lines = []
            for chan, axis in self.axes.iteritems():
                line = SpykeLine(self.x_vals,
                                 spike.data[chan],
                                 linewidth=0.005,
                                 color=colours[chan],
                                 antialiased=False)
                line._visible = False
                line.colour = colour
                axis.add_line(line)
                axis.autoscale_view()
                axis.set_ylim(self.yrange)
                line._visible = True
                lines.append(line)
            self.spikes[(spike, colour)] = [lines, True]

        self.draw(True)

    def remove(self, spike, colour):
        """ Remove the selected spike from the plot display. """
        self._toggleVisible(spike, colour)
        self.draw(True)

class ClickableSortPanel(OneAxisSortPanel):
    def __init__(self, *args, **kwargs):
        OneAxisSortPanel.__init__(self, *args, **kwargs)
        self.Bind(wx.EVT_LEFT_DCLICK, self.onDoubleClick, self)
        self.Bind(wx.EVT_LEFT_DOWN, self.onClick, self)
        #self.mpl_connect('button_press_event', self.onLeftDown)

    def _sendEvent(self, coords):
        event = ClickedChannelEvent(myEVT_CLICKED_CHANNEL, self.GetId())
        event.coords = coords
        self.GetEventHandler().ProcessEvent(event)

    def onDoubleClick(self, evt):
        coords = evt.GetPosition()
        self._sendEvent(coords)

    def onClick(self, evt):
        coords = evt.GetPosition()
        self._sendEvent(coords)

    def onLeftDown(self, event):
        a = event.inaxes
        b = AxesWrapper(a)
        chan = self.axesToChan[b]
        channels = [False] * len(self.axesToChan)
        channels[chan] = True
        print 'Sending event!'
        self._sendEvent(channels)


#####----- Tests

import spyke.detect
import os

filenames = ['C:\data\Cat 15\87 - track 7c spontaneous craziness.srf',
             'C:\Documents and Settings\Reza Lotun\Desktop\Surfdata\87 - track 7c spontaneous craziness.srf',
             '/media/windows/Documents and Settings/Reza ' \
                        'Lotun/Desktop/Surfdata/' \
                        '87 - track 7c spontaneous craziness.srf',
             '/home/rlotun/data_spyke/'\
                        '87 - track 7c spontaneous craziness.srf',
             '/data/87 - track 7c spontaneous craziness.srf',
             '/Users/rlotun/work/spyke/data/smallSurf',
             '/home/rlotun/spyke/data/smallSurf',
            ]

class Opener(object):
    def __init__(self):

        for filename in filenames:
            try:
                stat = os.stat(filename)
                break
            except:
                continue

        import spyke.surf
        surf_file = spyke.surf.File(filename)
        self.surf_file = surf_file
        surf_file.parse()
        self.dstream = spyke.stream.Stream(surf_file.highpassrecords)
        layout_name = surf_file.layoutrecords[0].electrode_name
        import spyke.layout
        self.layout = eval('spyke.layout.Polytrode' + layout_name[-3:])()
        self.curr = self.dstream.records[0].TimeStamp


class TestWindows(wx.App):
    def __init__(self, *args, **kwargs):
        kwargs['redirect'] = False
        wx.App.__init__(self, *args, **kwargs)

    def OnInit(self):
        op = Opener()
        self.events = panel = TestOneAxisEventWin(None, -1, 'Data', op,
                                                            size=(200,900))
        self.sort = panel3 = TestSortWin(None, -1, 'Events', op,
                                                            size=(200,900))
        self.chart = panel4 = TestOneAxisChartWin(None, -1, 'Chart One Axis', op,
                                                            size=(500,600))
        self.SetTopWindow(self.events)
        self.events.Show(True)
        self.sort.Show(True)
        self.chart.Show(True)

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
        #simp = spyke.detect.SimpleThreshold(self.stream,
        #                                    self.stream.records[0].TimeStamp)

        simp = spyke.detect.MultiPhasic(self.stream,
                                            self.stream.records[0].TimeStamp)
        self.event_iter = iter(simp)

        self.plotPanel = SortPanel(self, self.layout.SiteLoc)
        #self.plotPanel = OneAxisEventPanel(self, self.layout.SiteLoc)
        #self.plotPanel = ManyAxisSortPanel(self, self.layout.SiteLoc)

        #self.data = None
        #self.points = []
        #self.selectionPoints = []
        self.borderAxes = None

        self.timer.Start(200)

    def onTimerEvent(self, evt):
        #waveforms = self.dstream[self.curr:self.curr+self.incr]
        #self.curr += self.incr
        waveforms = self.event_iter.next()
        self.plotPanel.plot(waveforms)


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

class TestOneAxisEventWin(PlayWin):
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)

        self.plotPanel = OneAxisEventPanel(self, self.layout.SiteLoc)

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
        self.timer.Start(100)

    def onTimerEvent(self, evt):
        waveforms = self.stream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(waveforms)


class TestOneAxisChartWin(PlayWin):
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)
        self.plotPanel = OneAxisChartPanel(self, self.layout.SiteLoc)

        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = op.curr
        self.incr = 5000
        self.timer.Start(100)

    def onTimerEvent(self, evt):
        waveforms = self.stream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(waveforms)


if __name__ == '__main__':
    app = TestWindows()
    app.MainLoop()

