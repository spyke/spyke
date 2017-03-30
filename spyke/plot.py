"""Panels with embedded mpl figures based on FigureCanvasQTAgg.
Everything is plotted in units of uV and us"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
from copy import copy
import random
import numpy as np

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from matplotlib import rcParams
rcParams['lines.linestyle'] = '-'
rcParams['lines.marker'] = ''

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.mlab import poly_between

from core import MICRO, hex2rgb, toiter, intround

RED = '#ff0000'
ORANGE = '#ff7f00'
YELLOW = '#ffff00'
GREEN = '#00ff00'
CYAN = '#00ffff'
LIGHTBLUE = '#007fff'
BLUE = '#0000ff'
VIOLET = '#9f3fff' # pure violet (7f00ff) is a little too dark on a black background
MAGENTA = '#ff00ff'
GREY = '#555555'
WHITE = '#ffffff'
BROWN = '#af5050'
DARKGREY = '#303030'
LIGHTBLACK = '#202020'

SPIKELINEWIDTH = 1 # in points
SPIKELINESTYLE = '-'
NEURONLINEWIDTH = 1.5
NEURONLINESTYLE = '-'
ERRORALPHA = 0.15
RASTERLINEWIDTH = 1
RASTERLINESTYLE = '-'
TREFANTIALIASED = True
TREFLINEWIDTH = 0.5
TREFCOLOUR = DARKGREY
VREFANTIALIASED = True
VREFLINEWIDTH = 0.5
SELECTEDVREFLINEWIDTH = 3
VREFCOLOUR = DARKGREY
VREFSELECTEDCOLOUR = GREEN
SCALE = 1000, 100 # scalebar size in (us, uV)
SCALEXOFFSET = 25
SCALEYOFFSET = 15
SCALELINEWIDTH = 2
SCALECOLOUR = WHITE
CARETCOLOUR = LIGHTBLACK
CHANVBORDER = 175 # uV, vertical border space between top and bottom chans and axes edge

BACKGROUNDCOLOUR = 'black'

PLOTCOLOURS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA,
               GREY, WHITE, BROWN]
CLUSTERCOLOURS = copy(PLOTCOLOURS)
CLUSTERCOLOURS.remove(GREY)

CLUSTERCOLOURSRGB = hex2rgb(CLUSTERCOLOURS)
GREYRGB = hex2rgb([GREY])[0] # pull it out of the list

NCLOSESTCHANSTOSEARCH = 10
PICKRADIUS = 15 # required for 'line.contains(event)' call
#PICKTHRESH = 2.0 # in pixels? has to be a float or it won't work?

DEFNPLOTS = 200 # default number of plots to init in SortPanel
DEFNFILLS = 50 # default number of fills to init in SortPanel

CARETZORDER = 0 # layering
TREFLINEZORDER = 1
VREFLINEZORDER = 2
SCALEZORDER = 3
ERRORZORDER = 4
PLOTZORDER = 5
RASTERZORDER = 6


class ColourDict(dict):
    """Just an easy way to cycle through colours given some index,
    like say a chan id or a neuron id. Better than using a generator,
    cuz you don't need to keep calling .next(). This is like a dict
    of infinite length"""
    def __init__(self, colours=None, nocolour=None):
        self.colours = colours
        self.nocolour = nocolour

    def __getitem__(self, key):
        if key < 1: # unclustered/multiunit
            return self.nocolour
        i = key % len(self.colours) - 1 # single unit nids are 1 based
        return self.colours[i]

    def __setitem__(self, key, val):
        raise RuntimeError('ColourDict is unsettable')


CLUSTERCOLOURDICT = ColourDict(colours=CLUSTERCOLOURS, nocolour=GREY)
CLUSTERCOLOURRGBDICT = ColourDict(colours=CLUSTERCOLOURSRGB, nocolour=GREYRGB)


class Plot(object):
    """Plot slot, holds a LineCollection of visible chans for plotting
    a single stretch of data, contiguous in time"""
    def __init__(self, chans, panel, visible=False):
        self.panel = panel # panel that self belongs to
        self.chans = chans # channels corresponding to current set of lines in LineCollection
        colors = [ self.panel.vcolours[chan] for chan in chans ]
        self.lc = LineCollection([], linewidth=SPIKELINEWIDTH, linestyle=SPIKELINESTYLE,
                                 colors=colors,
                                 zorder=PLOTZORDER,
                                 antialiased=True,
                                 visible=visible,
                                 pickradius=PICKRADIUS)
        self.panel.ax.add_collection(self.lc) # add to panel's axes' pool of LCs
        self.n = None # Neuron associated with this plot
        # string id that indexes into SortPanel.used_plots,
        # starts with 's' or 'n' for spike or neuron:
        self.id = None
        self.fill = None # associated Fill

    def update(self, wave, tref):
        """Update LineCollection segments data from wave, and associated Fill.
        It's up to the caller to update colours if needed"""
        self.tref = tref
        AD2uV = self.panel.AD2uV
        nchans, npoints = wave.data.shape
        segments = np.zeros((nchans, npoints, 2)) # x vals in col 0, yvals in col 1
        data = AD2uV(wave.data) # convert AD wave data to uV
        if wave.ts is None: # or maybe check if data.size == 0 too
            x = []
            y = []
        else:
            x = np.tile(wave.ts-tref, nchans)
            x.shape = nchans, npoints
            segments[:, :, 0] = x
            segments[:, :, 1] = self.panel.gain * data
            # add offsets:
            for chani, chan in enumerate(wave.chans):
                xpos, ypos = self.panel.pos[chan]
                segments[chani, :, 0] += xpos
                segments[chani, :, 1] += ypos
        self.lc.set_segments(segments)
        self.chans = wave.chans
        if self.fill != None:
            self.fill.update(wave, tref)

    def show(self, enable=True):
        """Show/hide LC"""
        self.lc.set_visible(enable)
        if self.fill != None:
            if enable and not self.panel.enable_fills:
                # don't enable fill if global flag says to not do so
                return
            self.fill.show(enable)

    def hide(self):
        """Hide LC"""
        self.show(False)
        if self.fill != None:
            self.fill.hide()

    def visible(self):
        """Visibility status"""
        return self.lc.get_visible()

    def set_alpha(self, alpha):
        """Set alpha transparency for LC"""
        self.lc.set_alpha(alpha)

    def set_colours(self, colours):
        """Set colour(s) for LC"""
        self.lc.set_color(colours)
        if self.fill != None:
            self.fill.set_colours(colours)

    def update_colours(self):
        colours = [ self.panel.vcolours[chan] for chan in self.chans ]
        self.set_colours(colours)

    def set_stylewidth(self, style, width):
        """Set LC style and width"""
        self.lc.set_linestyle(style)
        self.lc.set_linewidth(width)

    def draw(self):
        """Draw LC to axes buffer (or whatever), avoiding unnecessary
        draws of all other artists in axes"""
        self.panel.ax.draw_artist(self.lc)
        if self.fill != None:
            self.fill.draw()


class Fill(object):
    """Fill slot, holds a PolyCollection of filled errors of visible neuron
    mean waveforms - only applicable in a SortPanel"""
    def __init__(self, chans, panel, visible=False):
        self.panel = panel # panel that self belongs to
        self.chans = chans # channels corresponding to current set of verts in PolyCollection
        self.pc = PolyCollection([],
                                 zorder=ERRORZORDER,
                                 alpha=ERRORALPHA,
                                 antialiased=True,
                                 visible=visible)
        self.panel.ax.add_collection(self.pc) # add to panel's axes' pool of PCs

    def update(self, wave, tref):
        """Update PolyCollection vertex data from wave. It's up to the caller to
        update colours if needed"""
        AD2uV = self.panel.AD2uV
        nchans, npoints = wave.std.shape
        # each timepoint has a +ve and a -ve vertex; x vals in col 0, yvals in col 1:
        verts = np.zeros((nchans, 2*npoints, 2))
        data = AD2uV(wave.data) # convert AD wave data to uV
        err = 2 * AD2uV(wave.std) # convert AD wave std to uV, double it for better visibility
        if wave.ts is None: # or maybe check if data.size == 0 too
            x = []
            y = []
        else:
            x = wave.ts-tref
            lower = self.panel.gain * (data - err)
            upper = self.panel.gain * (data + err)
            for chani, chan in enumerate(wave.chans):
                vert = poly_between(x, lower[chani], upper[chani])
                vert = np.asarray(vert).T
                # add offsets:
                xpos, ypos = self.panel.pos[chan]
                vert[:, 0] += xpos
                vert[:, 1] += ypos
                verts[chani] = vert
        self.pc.set_verts(verts)

    def show(self, enable=True):
        """Show/hide PC"""
        self.pc.set_visible(enable)

    def hide(self):
        """Hide PC"""
        self.show(False)

    def visible(self):
        """Visibility status"""
        return self.pc.get_visible()

    def set_colours(self, colours):
        """Set colour for PC, colours should really only ever be of length 1"""
        self.pc.set_color(colours)

    def draw(self):
        """Draw PC to axes buffer (or whatever), avoiding unnecessary
        draws of all other artists in axes"""
        self.panel.ax.draw_artist(self.pc)


class Rasters(object):
    """Holds a LineCollection of rasters corresponding
    to all spikes visible in panel, one vertical line per chan per spike"""
    def __init__(self, panel):
        self.panel = panel # panel that self belongs to
        self.lc = LineCollection([], linewidth=RASTERLINEWIDTH, linestyle=RASTERLINESTYLE,
                                 zorder=RASTERZORDER,
                                 antialiased=True,
                                 visible=True,
                                 pickradius=PICKRADIUS)
        self.panel.ax.add_collection(self.lc) # add to panel's axes' pool of LCs

    def update(self, spikes, tref):
        """Update LineCollection from spikes, using spike times and chans"""
        #self.spikes = spikes
        #self.tref = tref
        nsegments = spikes['nlockchans'].sum()
        # 2 points per raster line, x vals in col 0, yvals in col 1
        segments = np.zeros((nsegments, 2, 2))
        colours = np.zeros(nsegments, dtype='|S7') # colours are length-7 strings
        segmenti = 0
        for spike in spikes:
            nchans = spike['nlockchans']
            # colour segments according to each spike's max chan:
            colours[segmenti:segmenti+nchans] = self.panel.vcolours[spike['chan']]
            spikechans = spike['lockchans'][:nchans]
            for chan in spikechans:
                xpos, ypos = self.panel.pos[chan]
                x = spike['t'] - tref + xpos
                chanheight = self.panel.RASTERHEIGHT # uV, TODO: calculate this somehow
                ylims = ypos - chanheight/2, ypos + chanheight/2
                segments[segmenti, :, 0] = x, x
                segments[segmenti, :, 1] = ylims
                segmenti += 1
        self.lc.set_segments(segments)
        self.lc.set_color(list(colours))

    def show(self, enable=True):
        """Show/hide LC"""
        self.lc.set_visible(enable)

    def hide(self):
        """Hide LC"""
        self.show(False)

    def visible(self):
        """Visibility status"""
        return self.lc.get_visible()

    def draw(self):
        """Draw LC to axes buffer (or whatever), avoiding unnecessary
        draws of all other artists in axes"""
        self.panel.ax.draw_artist(self.lc)


class PlotPanel(FigureCanvas):
    """A QtWidget with an embedded mpl figure. Base class for Data and Sort panels"""
    # not necessarily constants
    def __init__(self, parent, tw=None, cw=None):
        self.figure = Figure() # resize later? can also set dpi here
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.updateGeometry()

        self.spykewindow = parent.parent()

        self.available_plots = [] # pool of available Plots
        self.available_fills = [] # pool of available Fills
        # plots holding currently displayed spikes/neurons, indexed by sid/nid
        # with s or n prepended:
        self.used_plots = {}
        self.qrplt = None # current quickly removable Plot with associated .background
        self.rasters = None # Rasters object

        self.tw = tw # temporal window of each channel, in plot units (us ostensibly)
        self.cw = cw # temporal window of caret, in plot units

        self.figure.set_facecolor(BACKGROUNDCOLOUR)
        # should really just turn off the edge line altogether, but how?
        self.figure.set_edgecolor(BACKGROUNDCOLOUR)
        #self.figure.set_frameon(False) # not too sure what this does, causes painting problems

        self.mpl_connect('button_press_event', self.OnButtonPress) # bind figure mouse click
        # TODO: mpl is doing something weird that prevents arrow key press events:
        #self.mpl_connect('key_press_event', self.OnKeyPress)
        # happens when an artist with a .picker attrib has a mouse event happen within
        # epsilon distance of it:
        #self.mpl_connect('pick_event', self.OnPick)
        self.mpl_connect('motion_notify_event', self.OnMotion) # mouse motion within figure
        #self.mpl_connect('scroll_event', self.OnMouseWheel)

        if isinstance(self, SortPanel):
            probe = self.sort.probe # sort must exist by now if sort panel is being used
        else: # it's a Spike/Chart/LFP panel
            probe = self.stream.probe
        self.probe = probe
        self.SiteLoc = probe.SiteLoc # probe site locations with origin at center top
        self.chans = probe.SiteLoc.keys()
        self.chans.sort() # a sorted list of chans, keeps from having to do this over and over
        self.nchans = probe.nchans
        self.chans_selected = [] # for clustering, or potentially other uses as well

        # for mpl, convert probe SiteLoc to center bottom origin instead of center top
        siteloc = copy(self.SiteLoc) # lowercase means bottom origin
        ys = [ y for x, y in siteloc.values() ]
        maxy = max(ys)
        for key, (x, y) in siteloc.iteritems():
            y = maxy - y
            siteloc[key] = (x, y) # update
        self.siteloc = siteloc # center bottom origin

        self.init_axes()
        self.pos = {} # positions of line centers, in plot units (us, uV)
        self.vcolours = {} # colour mapping that cycles vertically in space
        self.do_layout() # defined by subclasses, sets self.pos
        self.xy_um = self.get_xy_um()
        x = self.xy_um[0]
        self.colxs = np.unique(x) # unique x values that demarcate columns
        self.colxs.sort() # guarantee they're in order from left to right
        self.ax.set_axis_off() # turn off the x and y axis
        self.ax.set_visible(True)
        self.ax.set_autoscale_on(False) # TODO: not sure if this is necessary
        self.draw()
        self.black_background = self.copy_from_bbox(self.ax.bbox) # init

        # add reference lines and caret in layered order
        self._show_tref(True) # call the _ methods directly, to prevent unnecessary draws
        self._show_vref(True)
        self._show_scale(True)
        self._show_caret(True)
        for ref in ['TimeRef', 'VoltageRef', 'Scale', 'Caret']:
            # enforce menu item toggle state
            self.spykewindow.ui.__dict__['action%s' % ref].setChecked(True)
        self.draw() # do a full draw of the ref lines
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # init
        self.init_plots()
        self.init_rasters()

    def get_stream(self):
        return self.spykewindow.hpstream # overridden in LFPPanel

    stream = property(get_stream)

    def get_sort(self):
        return self.spykewindow.sort

    sort = property(get_sort)

    def get_tres(self):
        return self.stream.tres # overriden in SortPanel

    tres = property(get_tres)

    def get_AD2uV(self):
        # use the stream for all panel types except SortPanel
        return self.stream.converter.AD2uV

    AD2uV = property(get_AD2uV) # convenience for Plot objects to reference

    def resizeEvent(self, event):
        """Redraw refs and resave background after resizing"""
        FigureCanvas.resizeEvent(self, event)
        self.draw_refs()

    def init_axes(self):
        """Init the axes and ref lines"""
        self.ax = self.figure.add_axes([0, 0, 1, 1], # lbwh relative to figure?
                                       axisbg=BACKGROUNDCOLOUR,
                                       frameon=False,
                                       alpha=1.)

    def init_plots(self):
        """Create Plots for this panel"""
        chans = self.spykewindow.chans_enabled
        self.qrplt = Plot(chans=chans, panel=self, visible=True) # just one for this base class
        self.used_plots[0] = self.qrplt

    def init_rasters(self):
        """Init Rasters object"""
        self.rasters = Rasters(self)

    def add_ref(self, ref):
        """Helper method for external use"""
        if ref == 'TimeRef':
            self._add_tref()
        elif ref == 'VoltageRef':
            self._add_vref()
        elif ref == 'Scale':
            self._add_scale()
        elif ref == 'Caret':
            self._add_caret()
        else:
            raise ValueError, 'invalid ref: %r' % ref

    def show_ref(self, ref, enable=True):
        """Helper method for external use"""
        if ref == 'TimeRef':
            self._show_tref(enable)
        elif ref == 'VoltageRef':
            self._show_vref(enable)
        elif ref == 'Scale':
            self._show_scale(enable)
        elif ref == 'Caret':
            self._show_caret(enable)
        else:
            raise ValueError, 'invalid ref: %r' % ref
        self.draw_refs()

    def draw_refs(self):
        """Redraws all enabled reflines, resaves reflines_background"""
        plotvisibility = {} # mapping of currently shown plots to their visibility status
        for pltid, plt in self.used_plots.iteritems():
            plotvisibility[pltid] = plt.visible()
            plt.hide()
        self.show_rasters(False)
        self.draw() # only draw all enabled refs - defined in FigureCanvas
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # update
        self.background = None # no longer valid
        for pltid, plt in self.used_plots.iteritems():
            visible = plotvisibility[pltid]
            plt.show(visible) # re-show just the plots that were previously visible
            plt.draw()
        self.show_rasters(True)
        self.blit(self.ax.bbox)

    def _add_tref(self):
        """Add vertical time reference LineCollection, one line per probe column"""
        self.tlc = LineCollection([], linewidth=TREFLINEWIDTH,
                                  colors=TREFCOLOUR,
                                  zorder=TREFLINEZORDER,
                                  antialiased=TREFANTIALIASED,
                                  visible=False)
        self.ax.add_collection(self.tlc) # add to axes' pool of LCs
        self._update_tref()

    def _add_vref(self):
        """Add horizontal voltage reference LineCollection, one line per probe channel"""
        self.vlc = LineCollection([], linewidth=VREFLINEWIDTH,
                                  colors=VREFCOLOUR,
                                  zorder=VREFLINEZORDER,
                                  antialiased=VREFANTIALIASED,
                                  visible=False)
        self.ax.add_collection(self.vlc) # add to axes' pool of LCs
        self._update_vref()

    def _add_scale(self):
        """Add time and voltage "L" scale bar, as a LineCollection"""
        # left and bottom offsets fine tuned for SpikeSortPanel
        l, b = self.ax.get_xlim()[0] + SCALEXOFFSET, self.ax.get_ylim()[0] + SCALEYOFFSET
        tbar = (l, b), (l+SCALE[0], b) # us
        vbar = (l, b), (l, b+SCALE[1]*self.gain) # uV
        self.scale = LineCollection([tbar, vbar], linewidth=SCALELINEWIDTH,
                                    colors=SCALECOLOUR,
                                    zorder=SCALEZORDER,
                                    antialiased=True,
                                    visible=False)
        self.ax.add_collection(self.scale) # add to axes' pool of LCs

    def _add_caret(self):
        """Add a shaded rectangle to represent the time window shown in the spike frame"""
        ylim = self.ax.get_ylim()
        xy = (self.cw[0], ylim[0]) # bottom left coord of rectangle
        width = self.cw[1] - self.cw[0]
        height = ylim[1] - ylim[0]
        self.caret = Rectangle(xy, width, height,
                               facecolor=CARETCOLOUR,
                               zorder=CARETZORDER,
                               linewidth=0,
                               antialiased=False,
                               visible=False)
        self.ax.add_patch(self.caret)

    def _update_tref(self):
        """Update position and size of vertical time reference LineCollection"""
        # get column x positions:
        cols = np.unique([ xpos for (xpos, ypos) in self.pos.values() ])
        ylims = self.ax.get_ylim()
        nsegments = len(cols) # ie, ncols
        # 2 points per vertical tref line, x vals in col 0, yvals in col 1
        segments = np.zeros((nsegments, 2, 2))
        x = np.repeat(cols, 2)
        x.shape = nsegments, 2
        y = np.tile(ylims, nsegments)
        y.shape = nsegments, 2
        segments[:, :, 0] = x
        segments[:, :, 1] = y
        self.tlc.set_segments(segments)

    def _update_vref(self):
        """Update position and size of horizontal voltage reference LineCollection"""
        # somehow "chans, (xpos, ypos) = self.pos.items()" doesn't work...
        cxy = np.asarray([ (chan, xpos, ypos) for chan, (xpos, ypos) in self.pos.items() ])
        chans, xpos, ypos = cxy[:, 0], np.float64(cxy[:, 1]), np.float64(cxy[:, 2])
        self.chan2vrefsegmenti = {}
        for segmenti, chan in enumerate(chans):
            self.chan2vrefsegmenti[chan] = segmenti
        nsegments = len(chans)
        # 2 points per horizontal vref line, x vals in col 0, yvals in col 1
        segments = np.zeros((nsegments, 2, 2))
        x = np.repeat(xpos, 2)
        extra = (self.tw[1] - self.tw[0]) / 20 # vref horizontal overhang
        endbit = self.tres # the width of one timepoint
        x[0::2] += self.tw[0] - extra # left edge of each vref
        x[1::2] += self.tw[1] + extra - endbit # right edge of each vref
        x.shape = nsegments, 2
        y = np.repeat(ypos, 2) # y vals are the same for left and right edge of each vref
        y.shape = nsegments, 2
        segments[:, :, 0] = x
        segments[:, :, 1] = y
        self.vlc.set_segments(segments)
        self.segments = segments # save for potential later use

    def _update_scale(self):
        """Update scale bar position and size, based on current axes limits"""
        l, b = self.ax.get_xlim()[0] + SCALEXOFFSET, self.ax.get_ylim()[0] + SCALEYOFFSET
        tbar = (l, b), (l+SCALE[0], b) # us
        vbar = (l, b), (l, b+SCALE[1]*self.gain) # uV
        self.scale.set_segments([tbar, vbar])

    def _update_caret_width(self):
        """Update caret width"""
        self.caret.set_x(self.cw[0]) # bottom left coord of rectangle
        self.caret.set_width(self.cw[1]-self.cw[0])

    def _show_tref(self, enable=True):
        try:
            self.tlc
        except AttributeError:
            self._add_tref()
        self.tlc.set_visible(enable)

    def _show_vref(self, enable=True):
        try:
            self.vlc
        except AttributeError:
            self._add_vref()
        self.vlc.set_visible(enable)

    def _show_scale(self, enable=True):
        try:
            self.scale
        except AttributeError:
            self._add_scale()
        self.scale.set_visible(enable)

    def _show_caret(self, enable=True):
        try:
            self.caret
        except AttributeError:
            self._add_caret()
        self.caret.set_visible(enable)

    def set_chans(self, chans):
        """Reset chans for this plot panel, triggering colour update"""
        self.qrplt.chans = chans
        self.qrplt.update_colours()

    def get_spatialchans(self, order='vertical'):
        """Return channels in spatial order.
        order='vertical': sort from bottom to top, left to right
        order='horziontal': sort from left to right, bottom to top
        TODO: fix code duplication"""
        if order == 'vertical':
            # first, sort x coords, then y: (secondary, then primary)
            xychans = [ (x, y, chan) for chan, (x, y) in self.siteloc.iteritems() ] # list of (x, y, chan) 3-tuples
            xychans.sort() # stable sort in-place according to x values (first in tuple)
            yxchans = [ (y, x, chan) for (x, y, chan) in xychans ]
            yxchans.sort() # stable sort in-place according to y values (first in tuple)
            chans = [ chan for (y, x, chan) in yxchans ] # unload the chan indices, now sorted bottom to top, left to right
        elif order == 'horizontal':
            # first, sort y coords, then x: (secondary, then primary)
            yxchans = [ (y, x, chan) for chan, (x, y) in self.siteloc.iteritems() ] # list of (y, x, chan) 3-tuples
            yxchans.sort() # stable sort in-place according to y values (first in tuple)
            xychans = [ (x, y, chan) for (y, x, chan) in yxchans ] # list of (x, y, chan) 3-tuples
            xychans.sort() # stable sort in-place according to x values (first in tuple)
            chans = [ chan for (x, y, chan) in xychans ] # unload the chan indices, now sorted left to right, bottom to top
        else:
            raise ValueError
        return chans

    def get_xy_um(self):
        """Pull xy tuples in um out of self.pos, store in (2 x nchans) array,
        in self.chans order. In chart and lfp panels, this is different from siteloc,
        since these panels have only a single column"""
        xy_um = np.asarray([ (self.us2um(self.pos[chan][0]), self.uv2um(self.pos[chan][1]))
                                  for chan in self.chans ]).T # x is row0, y is row1
        return xy_um
    '''
    def get_spatial_chan_array(self):
        """build up 2d array of chanis, with cols and rows sorted according to probe layout"""
        x, y = self.xy_um # in self.chans order
        # should correspond to unique columns in spike frame, or just single column in chart or lfp frame
        a = []
        uniquexs = set(x)
        for uniquex in uniquexs:
            x == uniquexs
    '''
    def get_closestchans(self, evt, n=1):
        """Return n channels in column closest to mouse event coords,
        sorted by vertical distance from mouse event"""

        # sum of squared distances
        #d2 = (x-xdata)**2 + (y-ydata)**2
        #i = d2.argsort()[:n] # n indices sorted from smallest squared distance to largest

        # what column is this event closest to? pick that column,
        # and then the n vertically closest chans within it
        xdata = self.us2um(evt.xdata) # convert mouse event to um
        ydata = self.uv2um(evt.ydata)
        x, y = self.xy_um
        # find nearest column
        dx = np.abs(xdata - self.colxs) # array of x distances
        coli = dx.argmin() # index of column nearest to mouse click
        colx = self.colxs[coli] # x coord of nearest column
        i, = (x == colx).nonzero() # indices into self.chans of chans that are in the nearest col
        colchans = np.asarray(self.chans)[i] # channels in nearest col
        dy = np.abs(y[i] - ydata) # vertical distances between mouse click and all chans in this col
        i = dy.argsort()[:n] # n indices sorted from smallest to largest y distance
        chans = colchans[i] # index into channels in the nearest column
        if len(chans) == 1:
            chans = chans[0] # pull it out, return a single value
        return chans

    def get_closestline(self, evt):
        """Return line that's closest to mouse event coords"""
        d2s = [] # sum squared distances
        hitlines = []
        closestchans = self.get_closestchans(evt, n=NCLOSESTCHANSTOSEARCH)
        for chan in closestchans:
            line = self.qrplt.lines[chan]
            if line.get_visible(): # only consider lines that are visible
                hit, tisdict = line.contains(evt)
                if hit:
                    tis = tisdict['ind'] # pull them out of the dict
                    xs = line.get_xdata()[tis]
                    ys = line.get_ydata()[tis]
                    d2 = (xs-evt.xdata)**2 + (ys-evt.ydata)**2
                    d2 = d2.min() # point on line closest to mouse
                    hitlines.append(line)
                    d2s.append(d2)
        d2s = np.asarray(d2s)
        if d2s.size != 0:
            linei = d2s.argmin() # index of line with smallest d2
            return hitlines[linei]
        else:
            return None

    def plot(self, wave, tref=None):
        """Plot waveforms and optionally rasters wrt a reference time point"""
        if tref == None:
            tref = wave.ts[0] # use the first timestamp in the waveform as the reference time point
        self.restore_region(self.reflines_background)
        # update plots and rasters
        self.qrplt.update(wave, tref)
        self.qrplt.draw()
        if self.spykewindow.ui.actionRasters.isChecked() and self.rasters != None:
            self.update_rasters(tref)
            self.rasters.draw()
        self.blit(self.ax.bbox)
        #self.gui_repaint()
        #self.draw()
        #self.Refresh() # possibly faster, but adds a lot of flicker

    def update_rasters(self, tref):
        """Update spike raster positions and visibility wrt tref"""
        try:
            s = self.sort
            s.spikes
        except AttributeError: return # no sort/spikes exist yet
        # find out which spikes are within time window:
        lo, hi = s.spikes['t'].searchsorted((tref+self.tw[0], tref+self.tw[1]))
        spikes = s.spikes[lo:hi] # spikes within range of current time window
        self.rasters.update(spikes, tref)

    def show_rasters(self, enable=True):
        """Show/hide all rasters in this panel"""
        if self.rasters != None:
            self.rasters.show(enable)
    '''
    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        self.spykewindow.usperum /= x
        self.update_tw(self.tw[0]/x, self.tw[1]/x) # scale time window endpoints
    '''
    def update_tw(self, tw):
        """Update tw and everything that depends on it"""
        self.tw = tw
        self.do_layout() # resets axes lims and recalcs self.pos
        self._update_tref()
        self._update_vref()
        self._update_scale()
        self.draw_refs()
        #self.post_motion_notify_event() # forces tooltip update, even if mouse hasn't moved
    '''
    def post_motion_notify_event(self):
        """Posts a motion_notify_event to mpl's event queue"""
        # get mouse pos relative to this window:
        x, y = wx.GetMousePosition() - self.GetScreenPosition()
        # now just mimic what mpl FigureCanvasWx._onMotion does
        y = self.figure.bbox.height - y
        # no wx event to pass as guiEvent:
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=None)
    '''
    def um2uv(self, um):
        """Vertical conversion from um in channel siteloc
        space to uV in signal space"""
        return self.spykewindow.uVperum * um

    def uv2um(self, uV):
        """Convert from uV to um"""
        return uV / self.spykewindow.uVperum

    def um2us(self, um):
        """Horizontal conversion from um in channel siteloc
        space to us in signal space"""
        return self.spykewindow.usperum * um

    def us2um(self, us):
        """Convert from us to um"""
        return us / self.spykewindow.usperum

    '''
    def OnNavigation(self, evt):
        """Navigation key press"""
        #print('nagivation event:', evt)
        if evt.GetDirection(): # forward
            direction = 1
        else: # backward
            direction = -1
        self.spykewindow.seek(self.qrplt.tref + direction*self.stream.tres)
        evt.Skip() # allow left, right, pgup and pgdn to propagate to OnKeyDown handler
    '''
    def OnButtonPress(self, evt):
        """Seek to timepoint as represented on chan closest to left click.
        Toggle specific chans on right click. On ctrl+left click, reset primary
        peak timepoint and maxchan of currently selected spike. On ctrl+right click,
        reset secondary peak timepoint"""
        spw = self.topLevelWidget().parent() # spyke window
        button = evt.button
        modifiers = QtGui.QApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
        # NOTE: evt.key is supposed to give us the modifier, if any (like ctrl or shift)
        # but doesn't seem to work in MPL in qt. Also, evt.guiEvent always seems to be
        # None in qt. Also, up and down scroll events don't work.
        chan = self.get_closestchans(evt, n=1)
        # find clicked timepoint:
        xpos = self.pos[chan][0]
        # undo position correction and convert from relative to absolute time:
        t = evt.xdata - xpos + self.qrplt.tref
        t = spw.get_nearest_timepoint(t)
        if ctrl:
            if button == 1: # left click
                # set t as primary peak and align selected spike to it, set maxchan
                self.alignselectedspike('primary', t, chan)
                self.spykewindow.seek(t) # seek to t
            elif button == 3: # right click
                # designate t as secondary peak of selected spike
                self.alignselectedspike('secondary', t)
                self.spykewindow.seek(t) # seek to t
        else:
            if button == 1: # left click
                self.spykewindow.seek(t) # seek to t
            elif button == 3: # right click
                # toggle closest chan
                if chan not in self.spykewindow.chans_enabled:
                    enable = True
                else:
                    enable = False
                self.spykewindow.set_chans_enabled(chan, enable) # this calls self.set_chans()

    def alignselectedspike(self, peaktype, t, chan=None):
        """Align spike selected in sortwin to t, where t is designated as the
        primary or secondary peak timepoint. Also optionally set the maxchan
        of the spike to chan. Since this is happening in a DataWindow, it's safe
        to assume that a .srf or .track file is open"""
        #if srff not open:
        #    print("can't align selected spike without .srf file(s)")
        spw = self.topLevelWidget().parent() # spyke window
        spikes = spw.sort.spikes
        try:
            sid = spw.GetSpike()
        except RuntimeError, msg:
            print(msg)
            return
        if peaktype == 'primary':
            spw.primarypeakt = t
        elif peaktype == 'secondary':
            spw.secondarypeakt = t
        if chan != None:
            nchans = spikes[sid]['nchans']
            chans = spikes[sid]['chans'][:nchans]
            if chan in chans:
                spw.alignspike2chan = chan
            else:
                print('ERROR: selected maxchan not in spikes chanlist')

    def reloadSelectedSpike(self):
        """Reload selected spike according to primary and secondary
        peak alignments previously set by clicking on raw waveform in self"""
        spw = self.spykewindow
        sort = spw.sort
        spikes = sort.spikes
        AD2uV = sort.converter.AD2uV
        try:
            sid = spw.GetSpike()
        except RuntimeError, msg:
            print(msg)
            return
        abort = False
        try:
            if spw.primarypeakt == None or spw.secondarypeakt == None:
                abort = True
        except AttributeError:
            abort = True
        if abort:
            print("new primary and secondary peaks need to be set before "
                  "reloading selected spike")
            return
        t = spw.primarypeakt
        spikes[sid]['t'] = t # us
        nchans = spikes[sid]['nchans']
        chans = spikes[sid]['chans'][:nchans]
        try:
            if spw.alignspike2chan != None:
                # update chan and chani:
                chan = spw.alignspike2chan
                assert chan in chans
                spikes[sid]['chan'] = chan
                spikes[sid]['chani'] = chans.searchsorted(chan) # chans are always sorted
        except AttributeError:
            pass
        t0 = t + sort.tw[0]
        t1 = t + sort.tw[1]
        spikes[sid]['t0'] = t0 # us
        spikes[sid]['t1'] = t1
        wave = spw.hpstream(t0, t1, chans)
        sort.wavedata[sid][:nchans] = wave.data
        assert t != spw.secondarypeakt
        if t < spw.secondarypeakt:
            aligni = 0
            peak0ti = wave.ts.searchsorted(t)
            peak1ti = wave.ts.searchsorted(spw.secondarypeakt)
        else:
            aligni = 1
            peak0ti = wave.ts.searchsorted(spw.secondarypeakt)
            peak1ti = wave.ts.searchsorted(t)
        # TODO: redo spatial localization.
        # For now, cheat and make peaktis the same for all chans:
        spikes[sid]['tis'][:nchans] = peak0ti, peak1ti
        spikes[sid]['dt'] = abs(spw.secondarypeakt - t) # us
        chani = spikes[sid]['chani']
        V0 = AD2uV(wave.data[chani, peak0ti]) # uV
        V1 = AD2uV(wave.data[chani, peak1ti])
        spikes[sid]['V0'] = V0
        spikes[sid]['V1'] = V1
        spikes[sid]['Vpp'] = abs(V1 - V0)

        # mark sid as dirty in .wave file
        spw.update_dirtysids([sid])

        # reset for next alignment session
        spw.primarypeakt = None
        spw.secondarypeakt = None
        spw.alignspike2chan = None
        
        # reload spike in sort panel
        sortwin = spw.windows['Sort']
        sortwin.panel.updateAllItems()
        
        # seek to new timepoint, this also automatically updates the raster line
        spw.seek(t)
        print('realigned and reloaded spike %d to t=%d on chan %d' % (sid, t, chan))
        
    '''
    def OnPick(self, evt):
        """Pop up a tooltip when mouse is within PICKTHRESH of a line"""
        tooltip = self.GetToolTip()
        if evt.mouseevent.inaxes:
            line = evt.artist # assume it's one of our SpykeLines, since those are the only ones with their .picker attrib enabled
            chan = line.chan
            xpos, ypos = self.pos[chan]
            t = evt.mouseevent.xdata - xpos + self.qrplt.tref # undo position correction and convert from relative to absolute time
            v = (evt.mouseevent.ydata - ypos) / self.gain
            if t >= self.stream.t0 and t <= self.stream.t1: # in bounds
                t = int(round(t / self.stream.tres)) * self.stream.tres # round to nearest (possibly interpolated) sample
                tip = 'ch%d\n' % chan + \
                      't=%d %s\n' % (t, MICRO+'s') + \
                      'V=%.1f %s\n' % (v, MICRO+'V') + \
                      'window=(%.3f, %.3f) ms' % (self.tw[0]/1000, self.tw[1]/1000)
                tooltip.SetTip(tip)
                tooltip.Enable(True)
            else: # out of bounds
                tooltip.Enable(False)
        else:
            tooltip.Enable(False)
    '''
    def OnMotion(self, evt):
        """Pop up a tooltip when figure mouse movement is over axes"""
        if not evt.inaxes:
            self.setToolTip('')
            return
        # or, maybe better to just post a pick event, and let the pointed to chan
        # (instead of clicked chan) stand up for itself
        chan = self.get_closestchans(evt, n=1)
        sortpanel = isinstance(self, SortPanel)
        if not sortpanel and (chan not in self.qrplt.chans):
            self.setToolTip('')
            return
        xpos, ypos = self.pos[chan]
        t = evt.xdata - xpos
        if not sortpanel:
            t += self.qrplt.tref
        v = (evt.ydata - ypos) / self.gain
        if sortpanel:
            try:
                tres = self.sort.stream.tres
            except AttributeError: # sort doesn't exist
                self.setToolTip('')
                return
        else:
            if not (t >= self.stream.t0 and t <= self.stream.t1): # out of bounds
                self.setToolTip('')
                return
            tres = self.stream.tres
        t = int(round(t / tres)) * tres # nearest sample
        tip = 'ch%d @ %r %s\n' % (chan, self.SiteLoc[chan], MICRO+'m') + \
              't=%d %s\n' % (t, MICRO+'s') + \
              'V=%.1f %s\n' % (v, MICRO+'V') + \
              'window=(%.3f, %.3f) ms' % (self.tw[0]/1000, self.tw[1]/1000)
        self.setToolTip(tip)

    def wheelEvent(self, event):
        """Scale voltage or time, or step left or right"""
        SCALEX = 5
        spw = self.spykewindow
        winclass2wintype = {SpikePanel: 'Spike', ChartPanel: 'Chart', LFPPanel: 'LFP'}
        wintype2wintw = {'Spike': 'spiketw', 'Chart': 'charttw', 'LFP': 'lfptw'}
        wintype = winclass2wintype[type(self)]
        modifiers = event.modifiers()
        ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
        shift = modifiers == Qt.ShiftModifier # only modifier is shift
        # event.delta() seems to always be a multiple of 120 for some reason:
        di = event.delta() / 120
        sign = np.sign(di)
        absdi = abs(di)
        if ctrl: # scale voltage
            if sign == 1:
                self.gain = self.gain * (1 + absdi / SCALEX)
            else:
                self.gain = self.gain / (1 + absdi / SCALEX)
        elif shift: # scale time
            if sign == 1:
                tw = tuple([t / (1 + absdi / SCALEX) for t in self.tw])
            else:
                tw = tuple([t * (1 + absdi / SCALEX) for t in self.tw])
            self.update_tw(tw) # update Panel display tw
            spw.__dict__[wintype2wintw[wintype]] = tw # update spyke window's data fetch tw
        else: # step left/right on wheel up/down
            win = self.parent()
            win.step(-di)
        spw.plot(wintypes=[wintype])
        print('%s window gain=%g, tw=(%g, %g) ms' %
              (wintype, self.gain, self.tw[0]/1000, self.tw[1]/1000))


class SpikePanel(PlotPanel):
    """Spike panel. Presents a narrow temporal window of all channels
    laid out according to self.siteloc"""
    RASTERHEIGHT = 75 # uV, TODO: calculate this instead

    def __init__(self, *args, **kwargs):
        self.gain = 1.5
        PlotPanel.__init__(self, *args, **kwargs)

    def do_layout(self):
        # ordered left to right, bottom to top:
        self.hchans = self.get_spatialchans('horizontal')
        # ordered bottom to top, left to right
        self.vchans = self.get_spatialchans('vertical')
        #print('horizontal ordered chans in Spikepanel:\n%r' % self.hchans)
        # x origin is somewhere in between the xlimits. xlimits are asymmetric
        # if self.tw is asymmetric:
        self.ax.set_xlim(self.um2us(self.siteloc[self.hchans[0]][0]) + self.tw[0],
                         self.um2us(self.siteloc[self.hchans[-1]][0]) + self.tw[1])
        self.ax.set_ylim(self.um2uv(self.siteloc[self.vchans[0]][1]) - CHANVBORDER,
                         self.um2uv(self.siteloc[self.vchans[-1]][1]) + CHANVBORDER)
        colourgen = itertools.cycle(iter(PLOTCOLOURS))
        for chan in self.vchans:
            # chan order doesn't matter for setting .pos, but it does for setting .colours
            self.pos[chan] = (self.um2us(self.siteloc[chan][0]),
                              self.um2uv(self.siteloc[chan][1]))
            # assign colours so that they cycle vertically in space:
            self.vcolours[chan] = colourgen.next()
    '''
    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        PlotPanel._zoomx(self, x)
        # update main spyke frame so its plot calls send the right amount of data
        self.spykewindow.spiketw = self.tw
        self.spykewindow.frames['chart'].panel.cw = self.tw # update caret width
        self.spykewindow.frames['chart'].panel._update_caret_width()
        self.spykewindow.plot(frametypes='spike') # replot
    '''
    def _add_caret(self):
        """Disable for SpikePanel"""
        pass

    def _update_caret_width(self):
        """Disable for SpikePanel"""
        pass

    def _show_caret(self, enable=True):
        """Disable for SpikePanel"""
        pass


class ChartPanel(PlotPanel):
    """Chart panel. Presents all channels layed out vertically according
    to the vertical order of their site coords in .siteloc"""
    RASTERHEIGHT = 75 # uV, TODO: calculate this instead

    def __init__(self, *args, **kwargs):
        self.gain = 1.0
        PlotPanel.__init__(self, *args, **kwargs)

    def do_layout(self):
        """Sets axes limits and calculates self.pos"""
        self.vchans = self.get_spatialchans('vertical') # ordered bottom to top, left to right
        self.ax.set_xlim(0 + self.tw[0], 0 + self.tw[1]) # x origin at center
        miny = self.um2uv(self.siteloc[self.vchans[0]][1])
        maxy = self.um2uv(self.siteloc[self.vchans[-1]][1])
        # average vertical spacing between chans, in uV:
        ngaps = max(self.nchans-1, 1) # at least 1
        vspace = (maxy - miny) / ngaps
        self.ax.set_ylim(miny - CHANVBORDER, maxy + CHANVBORDER)
        colourgen = itertools.cycle(iter(PLOTCOLOURS))
        for chani, chan in enumerate(self.vchans):
            #self.pos[chan] = (0, self.um2uv(self.siteloc[chan][1])) # x=0 centers horizontally
            # x=0 centers horizontally, equal vertical spacing:
            self.pos[chan] = (0, chani*vspace)
            # assign colours so that they cycle vertically in space:
            self.vcolours[chan] = colourgen.next()
    '''
    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        PlotPanel._zoomx(self, x)
        # update main spyke frame so its plot calls send the right amount of data
        self.spykewindow.charttw = self.tw
        self.spykewindow.frames['lfp'].panel.cw = self.tw # update caret width
        self.spykewindow.frames['lfp'].panel._update_caret_width()
        self.spykewindow.plot(frametypes='chart') # replot
    '''
    def _add_vref(self):
        """Disable for ChartPanel"""
        pass

    def _update_vref(self):
        """Disable for ChartPanel"""
        pass

    def _show_vref(self, enable=True):
        """Disable for ChartPanel"""
        pass

    def _update_caret_width(self):
        """Set optimal paint method"""
        PlotPanel._update_caret_width(self)
        #self.draw() # can be quite slow
        self.Refresh() # can be faster, but adds flicker


class LFPPanel(ChartPanel):
    """LFP Panel"""
    def __init__(self, *args, **kwargs):
        self.gain = 1.0
        ChartPanel.__init__(self, *args, **kwargs)

    def init_plots(self):
        ChartPanel.init_plots(self)
        # LFPPanel needs to filter chans after initing plots
        # get it right for first qrplt.update() call:
        self.set_chans(self.spykewindow.chans_enabled)

    def init_rasters(self):
        """Disable for LFPPanel"""
        pass

    def get_stream(self):
        return self.spykewindow.lpstream # override ChartPanel(PlotPanel)'s hpstream

    stream = property(get_stream)

    def do_layout(self):
        """This is only necessary for SurfStream, which has a .layout attrib, and only to
        prevent plotting vref lines for chans that don't exist in the LFP"""
        ChartPanel.do_layout(self)
        # Need to get a list of keys, not an iterator, since self.pos dict can change size
        # during iteration:
        try: self.stream.layout
        except AttributeError: return
        for chan in list(self.pos):
            if chan not in self.stream.layout.chans:
                del self.pos[chan] # remove siteloc chans not in lowpassmultichan record
                try:
                    self.chans.remove(chan) # in place
                except ValueError: # already removed from list on previous do_layout() call
                    pass

    def set_chans(self, chans):
        """This is only necessary for SurfStream, which has a .layout attrib.
        Reset chans for this LFPPanel, triggering colour update.
        Take intersection of lpstream.layout.chans and chans_enabled,
        conserving order in lpstream.layout.chans"""
        ## TODO: LFP chan become incorrectly coloured for non-surf files when channels are
        ## disabled, probably has to do with this:
        try: self.stream.layout
        except AttributeError: return
        chans = [ chan for chan in self.stream.layout.chans if chan in chans ]
        ChartPanel.set_chans(self, chans)
    '''
    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        PlotPanel._zoomx(self, x)
        # update main spyke frame so its plot calls send the right amount of data
        self.spykewindow.lfptw = self.tw
        self.spykewindow.plot(frametypes='lfp') # replot
    '''
    def _add_vref(self):
        """Override ChartPanel"""
        PlotPanel._add_vref(self)

    def _update_vref(self):
        """Override ChartPanel"""
        PlotPanel._update_vref(self)

    def _show_vref(self, enable=True):
        """Override ChartPanel"""
        PlotPanel._show_vref(self, enable)

    def _update_caret_width(self):
        """Set optimal paint method"""
        PlotPanel._update_caret_width(self)
        self.draw()
        #self.Refresh() # possibly faster, but adds a lot of flicker


class SortPanel(PlotPanel):
    """A plot panel specialized for overplotting spikes and neurons"""
    def __init__(self, parent, tw=None):
        PlotPanel.__init__(self, parent, tw=tw)
        self.manual_selection = False
        self.maxed_out = False # has hit its maximum number of spike plots
        self.enable_fills = False # global enable flag for all fills
        self.sortwin = self.parent()

    def get_AD2uV(self):
        try: # use sort by default:
            return self.sort.converter.AD2uV
        except AttributeError: # sort doesn't exist yet
            return self.stream.converter.AD2uV

    AD2uV = property(get_AD2uV) # convenience for Plot objects to reference

    def get_tres(self):
        return self.sort.tres # override PlotPanel's definition

    tres = property(get_tres)

    def init_plots(self, nplots=DEFNPLOTS):
        """Add Plots to the pool of available ones"""
        for i in range(nplots): # Plots are init'd as invisible:
            plt = Plot(chans=self.chans, panel=self)
            self.available_plots.append(plt)

    def init_fills(self, nplots=DEFNFILLS):
        """Add Fills to the pool of available ones"""
        for i in range(nplots): # Fills are init'd as invisible:
            fill = Fill(chans=self.chans, panel=self)
            self.available_fills.append(fill)

    def init_rasters(self):
        """Disable for SortPanel"""
        pass

    def update_tw(self, tw):
        """Same as parent, but auto-refresh all plots after"""
        PlotPanel.update_tw(self, tw)
        self.updateAllItems()

    def _add_vref(self):
        """Increase pick radius for vrefs from default zero, since we're
        relying on them for tooltips"""
        PlotPanel._add_vref(self)
        self.vlc.set_pickradius(PICKRADIUS)

    def show_ref(self, ref, enable=True):
        PlotPanel.show_ref(self, ref, enable)
        self.qrplt = None
        self.background = None

    def addItems(self, items):
        """Add items (spikes/neurons) to self"""
        if items == []:
            return # do nothing
        if len(items) == 1:
            # before blitting this single item to screen, grab current buffer,
            # save as new background for quick restore if the next action is removal of
            # this very same item:
            self.background = self.copy_from_bbox(self.ax.bbox)
            # add the single item, save reference to its plot:
            self.qrplt = self.addItem(items[0])
            #print('saved quick remove plot %r' % self.qrplt)
        else:
            self.background = None
            # add all items
            for item in items:
                self.addItem(item)
        self.blit(self.ax.bbox)

    def addItem(self, item):
        """Put item in an available Plot, return the Plot"""
        s = self.sort
        if len(self.available_plots) == 0: # if we've run out of plots for additional items
            self.init_plots() # init another batch of plots
        plt = self.available_plots.pop() # pop a Plot to assign this item to
        plt.id = item
        id = int(item[1:])
        if item[0] == 'n': # it's a neuron
            if len(self.available_fills) == 0:
                # if we've run out of fills for additional neurons                
                self.init_fills() # init another batch of fills
            plt.fill = self.available_fills.pop() # pop a Fill, bind it to plot
            n = s.neurons[id]
            t = n.t
            colour = CLUSTERCOLOURDICT[id]
            alpha = 1
            style = NEURONLINESTYLE
            width = NEURONLINEWIDTH
            n.plt = plt # bind plot to neuron
            plt.n = n # bind neuron to plot
            wave = n.get_wave() # calls n.update_wave() if necessary
        else: # item[0] == 's' # it's a spike
            t = s.spikes['t'][id]
            nid = s.spikes['nid'][id]
            colour = CLUSTERCOLOURDICT[nid]
            alpha = 0.5
            if nid < 1:
                alpha = 0.75 # junk/multiunit GREY is just a bit too dark to leave at 0.5 alpha
            style = SPIKELINESTYLE
            width = SPIKELINEWIDTH
            wave = s.get_wave(id)
        plt.set_colours([colour])
        plt.set_alpha(alpha) # doesn't affect Fill alpha, if any
        plt.set_stylewidth(style, width)
        self.used_plots[plt.id] = plt # push it to the used plot stack
        # slice wave according to time window of this panel:
        wave = wave[t+self.tw[0] : t+self.tw[1]]
        # also updates, shows, and draws Fill, if applicable:
        plt.update(wave, t)
        plt.show()
        plt.draw()
        return plt

    def removeItems(self, items):
        """Remove items from plots
        TODO: set obj.wave = None when no longer plotting?"""
        if items == []: # do nothing
            return
        for item in items:
            # remove items from .used_plots, use contents of
            # .used_plots to decide how to do the actual plot removal
            plt = self.removeItem(item)
        # now remove items from actual plot
        if self.used_plots == {}:
            # restore blank background with just the ref lines:
            self.restore_region(self.reflines_background)
        elif len(items) == 1 and plt == self.qrplt and self.background != None:
            # remove the last added plot if a saved bg is available
            #print('quick removing plot %r' % self.qrplt)
            self.restore_region(self.background) # restore saved bg
        else:
            # remove more than one, but not all items
            # restore blank background with just the ref lines:
            self.restore_region(self.reflines_background)
            for plt in self.used_plots.values():
                plt.draw() # redraw the remaining plots in .used_plots
        # what was background is no longer useful for quick restoration on any other
        # item removal:
        self.background = None
        self.qrplt = None # qrplt set in addItems is no longer quickly removable
        self.blit(self.ax.bbox) # blit everything to screen

    def removeAllItems(self):
        """Shortcut for removing all items from plots"""
        items = [ plt.id for plt in self.used_plots.values() ]
        self.removeItems(items)

    def removeItem(self, item):
        """Restore item's Plot from used to available plot pool, return the Plot.
        Restore Fill as well, if applicable"""
        try:
            plt = self.used_plots.pop(item)
        except KeyError: # item isn't plotted, return None
            return
        # TODO: reset plot colour and line style here, or just set them each time in addItem?
        plt.id = None # clear its index into .used_plots
        if item[0] == 'n': # it's a neuron
            plt.n.plt = None # unbind plot from neuron
            fill = plt.fill
            fill.hide()
            plt.fill = None
            self.available_fills.append(fill)
        plt.hide() # hide all chan lines and fills
        self.available_plots.append(plt)
        return plt

    def showFills(self, enable=True):
        """Toggle visibility of all currently bound fills"""
        self.enable_fills = enable # update global flag
        self.restore_region(self.reflines_background)
        for item, plt in self.used_plots.items():
            if item[0] == 'n': # only neuron plots have fills
                plt.fill.show(enable)
            plt.draw() # redraw each plot with a bound fill
        # what was background is no longer useful for quick restoration on any other
        # item removal:
        self.background = None
        self.qrplt = None # qrplt set in addItems is no longer quickly removable
        self.blit(self.ax.bbox) # blit everything to screen

    def updateItems(self, items):
        """Re-plot items, potentially because their WaveForms have changed.
        Typical use case: spike is added to a neuron, neuron's mean waveform has changed"""
        if items == []: # do nothing
            return
        plt0 = self.used_plots[items[0]]
        if len(items) == 1 and plt0 != None and plt0 == self.qrplt and self.background != None:
            #print('quick removing and replotting plot %r' % self.qrplt)
            self.restore_region(self.background) # restore saved bg
            self.updateItem(items[0])
        else: # update and redraw all items
            # restore blank background with just the ref lines:
            self.restore_region(self.reflines_background)
            for item in items:
                self.updateItem(item)
            # what was background is no longer useful for quick restoration on any other
            # item removal:
            self.background = None
            self.qrplt = None # qrplt set in addItems is no longer quickly removable
        self.blit(self.ax.bbox) # blit everything to screen

    def updateAllItems(self):
        """Shortcut for updating all items in used_plots"""
        items = list(self.used_plots) # dict keys are plot ids
        self.updateItems(items)

    def updateItem(self, item):
        """Update and draw an item's plot"""
        s = self.sort
        plt = self.used_plots[item]
        id = int(item[1:])
        if item[0] == 'n': # it's a neuron
            n = s.neurons[id]
            t = n.t
            wave = n.get_wave() # calls n.update_wave() if necessary
        else: # item[0] == 's' # it's a spike
            t = s.spikes['t'][id]
            wave = s.get_wave(id)
        # slice wave according to time window of this panel:
        wave = wave[t+self.tw[0] : t+self.tw[1]]
        plt.update(wave, t)
        plt.draw()

    def get_closestline(self, evt):
        """Return line that's closest to mouse event coords, Slightly modified
        from PlotPanel's version"""
        d2s = [] # sum squared distances
        hitlines = []
        closestchans = self.get_closestchans(evt, n=NCLOSESTCHANSTOSEARCH)
        for chan in closestchans:
            line = self.vlines[chan] # consider all voltage ref lines, even if invisible
            hit, tisdict = line.contains(evt)
            if hit:
                tis = tisdict['ind'] # pull them out of the dict
                xs = line.get_xdata()[tis]
                ys = line.get_ydata()[tis]
                d2 = (xs-evt.xdata)**2 + (ys-evt.ydata)**2
                d2 = d2.min() # point on line closest to mouse
                hitlines.append(line)
                d2s.append(d2)
        d2s = np.asarray(d2s)
        if d2s.size != 0:
            linei = d2s.argmin() # index of line with smallest d2
            return hitlines[linei]
        else:
            return None

    def OnButtonPress(self, evt):
        """Toggle or clear channel selection for clustering by waveform shape, or for other
        potential uses as well"""
        button = evt.button
        if button == 1: # left click
            chan = self.get_closestchans(evt, n=1)
            if chan not in self.chans_selected: # it's unselected, select it
                self.chans_selected.append(chan)
            else: # it's selected, unselect it
                self.chans_selected.remove(chan)
            self.manual_selection = True
        elif button == 2: # middle click
            self.sortwin.spykewindow.ui.plotButton.click() # same as hitting ENTER in nslist
            #self.sortwin.on_actionSelectRandomSpikes_triggered()
        elif button == 3: # right click
            self.chans_selected = [] # clear channel selection
            self.manual_selection = False
        self.update_selvrefs()
        self.draw_refs() # update

    def update_selvrefs(self):
        """Set line widths, lengths, and colours of vrefs according to chans in
        self.chans_selected. Note that any changes to the display code here should also be
        made in the timepoint selection code for dimension reduction in
        SortWindow.get_tis()"""
        segments = self.segments.copy()
        colours = np.repeat(VREFCOLOUR, len(self.pos)) # clear all lines
        linewidths = np.repeat(VREFLINEWIDTH, len(self.pos)) # clear all lines
        inclt = self.sortwin.inclt
        # scale time selection around t=0 according to window asymmetry:
        dtw = self.sort.tw[1] - self.sort.tw[0] # spike time window width
        incltleft = intround(abs(self.tw[0]) / dtw * inclt) # left fraction wrt xpos
        incltright = inclt - incltleft # right fraction wrt xpos
        #print("self.tw, incltleft, incltright", self.tw, incltleft, incltright)
        for chan in self.chans_selected: # set line colour of selected chans
            vrefsegmenti = self.chan2vrefsegmenti[chan] # one vref for every enabled chan
            xpos = self.pos[chan][0] # chan xpos center (us)
            # modify the x values of this segment:
            x0, x1 = intround(xpos-incltleft), intround(xpos+incltright)
            segments[vrefsegmenti][:, 0] = x0, x1
            colours[vrefsegmenti] = VREFSELECTEDCOLOUR
            linewidths[vrefsegmenti] = SELECTEDVREFLINEWIDTH
        self.vlc.set_segments(segments)
        self.vlc.set_color(list(colours))
        self.vlc.set_linewidth(list(linewidths))


class SpikeSortPanel(SortPanel, SpikePanel):
    def __init__(self, parent, tw=None):
        self.gain = 1.5
        SortPanel.__init__(self, parent, tw=tw)

    def wheelEvent(self, event):
        """Scroll gainComboBox or incltComboBox on mouse wheel scroll"""
        modifiers = event.modifiers()
        ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
        sw = self.topLevelWidget() # SortWindow
        if ctrl: # scroll gainComboBox
            cbox = sw.gainComboBox
            on_box_triggered = sw.on_gainComboBox_triggered
        else: # scroll incltComboBox
            cbox = sw.incltComboBox
            on_box_triggered = sw.on_incltComboBox_triggered
        nitems = cbox.count()
        # event.delta() seems to always be a multiple of 120 for some reason:
        di = event.delta() / 120
        # both combo boxes are sorted in decreasing order, hence the negation of di:
        i = min(max(cbox.currentIndex()-di, 0), nitems-1)
        cbox.setCurrentIndex(i)
        on_box_triggered() # as if it were user selected

#class ChartSortPanel(SortPanel, ChartPanel):
