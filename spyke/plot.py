"""Panels with embedded mpl figures based on FigureCanvasQTAgg.
Everything is plotted in units of uV and us"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
from copy import copy
import random
import numpy as np

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from matplotlib import rcParams
rcParams['lines.linestyle'] = '-'
rcParams['lines.marker'] = ''

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PolyCollection

from .core import hex2rgb, toiter, intround, poly_between

RED = '#ff0000'
ORANGE = '#ff7f00'
YELLOW = '#ffff00'
GREEN = '#00ff00'
CYAN = '#00ffff'
LIGHTBLUE = '#007fff'
BLUE = '#0000ff'
VIOLET = '#9f3fff' # pure violet (7f00ff) is a little too dark on a black background
MAGENTA = '#ff00ff'
GREY = '#666666'
WHITE = '#ffffff'
BROWN = '#af5050'
DARKGREY = '#303030'
LIGHTBLACK = '#202020'

SPIKELINEWIDTH = 1 # in points
SPIKELINESTYLE = '-'
NEURONLINEWIDTH = 1.5
NEURONLINESTYLE = '-'
ERRORALPHA = 0.3
RASTERLINEWIDTH = 1
RASTERLINESTYLE = '-'
STIMSLINEWIDTH = 2
STIMSLINESTYLE = '-'
TREFANTIALIASED = True
TREFLINEWIDTH = 0.5
TREFCLR = DARKGREY
VREFANTIALIASED = True
VREFLINEWIDTH = 0.5
SELECTEDVREFLINEWIDTH = 3
VREFCLR = DARKGREY
VREFSELECTEDCLR = GREEN
SCALE = 1000, 100 # scalebar size in (us, uV)
SCALEXOFFSET = 25
SCALEYOFFSET = 15
SCALELINEWIDTH = 2
SCALECLR = WHITE
CARETCLR = LIGHTBLACK
CHANVBORDER = 175 # uV, vertical border space between top and bottom chans and axes edge

BACKGROUNDCLR = 'black'

PLOTCLRS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA,
            GREY, WHITE, BROWN]
CLUSTERCLRS = copy(PLOTCLRS)
CLUSTERCLRS.remove(GREY)

CLUSTERCLRSRGB = hex2rgb(CLUSTERCLRS)
GREYRGB = hex2rgb([GREY])[0] # pull it out of the list

NCLOSESTCHANSTOSEARCH = 10
PICKRADIUS = 15 # required for 'line.contains(event)' call
#PICKTHRESH = 2.0 # in pixels? has to be a float or it won't work?

CARETZORDER = 0 # layering
TREFLINEZORDER = 1
VREFLINEZORDER = 2
SCALEZORDER = 3
ERRORZORDER = 4
PLOTZORDER = 5
RASTERZORDER = 6
STIMSZORDER = 7


class ColorDict(dict):
    """Just an easy way to cycle through colors given some index,
    like say a chan id or a neuron id. Better than using a generator,
    because you don't need to keep calling next(). This is like a dict
    of infinite length"""
    def __init__(self, colors=None, nocolor=None):
        self.colors = colors
        self.nocolor = nocolor

    def __getitem__(self, key):
        if key < 1: # unclustered/multiunit
            return self.nocolor
        i = key % len(self.colors) - 1 # single unit nids are 1 based
        return self.colors[i]

    def __setitem__(self, key, val):
        raise RuntimeError('ColorDict is unsettable')


CLUSTERCLRDICT = ColorDict(colors=CLUSTERCLRS, nocolor=GREY)
CLUSTERCLRRGBDICT = ColorDict(colors=CLUSTERCLRSRGB, nocolor=GREYRGB)


class Plot(object):
    """Plot slot, holds a LineCollection of visible chans for plotting
    a single stretch of data, contiguous in time"""
    def __init__(self, chans, panel, visible=False):
        self.panel = panel # panel that self belongs to
        self.chans = chans # channels corresponding to current set of lines in LineCollection
        colors = [ self.panel.vcolors[chan] for chan in chans ]
        self.lc = LineCollection([],
                                 linewidth=SPIKELINEWIDTH,
                                 linestyle=SPIKELINESTYLE,
                                 colors=colors,
                                 zorder=PLOTZORDER,
                                 antialiased=True,
                                 visible=visible,
                                 pickradius=PICKRADIUS)
        self.panel.ax.add_collection(self.lc) # add to panel's axes' pool of LCs

    def update(self, wave, tref):
        """Update LineCollection segments data from wave. It's up to the caller to update
        colors if needed"""
        self.tref = tref
        panel = self.panel
        nchans, nt = wave.data.shape
        segments = np.zeros((nchans, nt, 2)) # x vals in col 0, yvals in col 1
        if wave.ts is not None: # or maybe check if wave.data.size != 0 too
            x = np.tile(wave.ts-tref, nchans)
            x.shape = nchans, nt
            segments[:, :, 0] = x
            segments[:, :, 1] = panel.gain * panel.AD2uV(wave.data)
            # add offsets:
            for chani, chan in enumerate(wave.chans):
                xpos, ypos = panel.pos[chan]
                segments[chani, :, 0] += xpos
                segments[chani, :, 1] += ypos
        self.lc.set_segments(segments)
        self.chans = wave.chans

    def show(self, enable=True):
        """Show/hide LC"""
        self.lc.set_visible(enable)

    def hide(self):
        """Hide LC"""
        self.show(False)

    def visible(self):
        """Visibility status"""
        return self.lc.get_visible()

    def set_alpha(self, alpha):
        """Set alpha transparency for LC"""
        self.lc.set_alpha(alpha)

    def set_colors(self, colors):
        """Set color(s) for LC"""
        self.lc.set_color(colors) # synonymous with .set_colors() in newer MPL versions

    def update_colors(self):
        colors = [ self.panel.vcolors[chan] for chan in self.chans ]
        self.set_colors(colors)

    def set_stylewidth(self, style, width):
        """Set LC style and width"""
        self.lc.set_linestyle(style)
        self.lc.set_linewidth(width)

    def draw(self):
        """Draw LC to axes buffer (or whatever), avoiding unnecessary
        draws of all other artists in axes"""
        self.panel.ax.draw_artist(self.lc)


class SpikePlot(Plot):
    """Plot slot, holds a LineCollection of visible chans for plotting
    multiple stretches of data, i.e. multiple overlapping multichannel spikes
    or neuron mean waveforms"""
    def __init__(self, panel, linewidth=None, linestyle=None, alpha=None, visible=False):
        self.panel = panel # panel that self belongs to
        self.lc = LineCollection([],
                                 linewidth=linewidth,
                                 linestyle=linestyle,
                                 alpha=alpha,
                                 zorder=PLOTZORDER,
                                 antialiased=True,
                                 visible=visible,
                                 pickradius=PICKRADIUS)
        self.panel.ax.add_collection(self.lc) # add to panel's axes' pool of LCs
        self.nsegments = 0
        self.fill = None # associated Fill

    def update(self, waves, trefs):
        """Update LineCollection segments data from waves. Each entry in waves is a 2D array
        representing a multichannel spike. It's up to the caller to update colors if needed.
        Also update any associated Fill"""
        panel = self.panel
        allsegments = []
        for wave, tref in zip(waves, trefs):
            nchans, nt = wave.data.shape
            segments = np.zeros((nchans, nt, 2)) # x vals in col 0, yvals in col 1
            if wave.ts is not None: # or maybe check if wave.data.size != 0 too
                if isinstance(panel, SortPanel) and panel.spykewindow.normButton.isChecked():
                    wave = self.norm_wave(wave)
                x = np.tile(wave.ts-tref, nchans) # nt*nchans 1D array
                x.shape = nchans, nt
                segments[:, :, 0] = x
                segments[:, :, 1] = panel.gain * panel.AD2uV(wave.data)
                # add offsets:
                for chani, chan in enumerate(wave.chans):
                    xpos, ypos = panel.pos[chan]
                    segments[chani, :, 0] += xpos
                    segments[chani, :, 1] += ypos
            allsegments.append(segments)
        # flatten into a list of nt x 2 arrays:
        segments = [ row for segment in allsegments for row in segment ]
        self.lc.set_segments(segments)
        self.nsegments = len(segments)
        if self.fill != None:
            self.fill.update(waves, trefs)

    def set_colors(self, colors):
        """Set color(s) for LC"""
        if len(colors) != self.nsegments:
            raise ValueError("Expected %d color values, one for each segment" % self.nsegments)
        self.lc.set_color(colors) # synonymous with .set_colors() in newer MPL versions
        if self.fill != None:
            self.fill.set_colors(colors)

    def norm_wave(self, wave):
        """Return wave with data normalized by Vpp of its max chan, subject to the current
        channel and timepoint selection"""
        panel = self.panel
        ti0, ti1 = panel.sortwin.get_tis()
        selchans = panel.chans_selected
        chans = np.intersect1d(wave.chans, selchans) # overlapping set
        if len(chans) == 0: # empty array, no overlap
            chans = wave.chans # ignore selected chans
        chanis = wave.chans.searchsorted(chans) # indices into data rows
        seldata = wave.data[chanis, ti0:ti1] # selected part of the waveform
        Vpp = seldata.ptp(axis=1).max() # Vpp chan with biggest Vpp
        # For display, scale up by Vpp of the highest amplitude plotted neuron.
        # This makes multiple neuron mean waveforms the same amplitude when plotted
        # simultaneously, allowing visual comparison purely by shape:
        neurons = [ panel.sort.neurons[nid] for nid in panel.nids ] # neurons currently plotted
        scales = []
        for neuron in neurons:
            nwave = neuron.wave
            chans = np.intersect1d(nwave.chans, selchans) # overlapping set
            if len(chans) == 0: # empty array, no overlap
                chans = nwave.chans # ignore selected chans
            chanis = nwave.chans.searchsorted(chans) # indices into data rows
            selndata = nwave.data[chanis, ti0:ti1] # selected part of the waveform
            scale = selndata.ptp(axis=1).max()
            scales.append(scale)
        if scales: # non-empty
            scale = max(scales)
        else: # no neuron plotted
            scale = 150 # arbitrary value to scale to
        if Vpp == 0:
            Vpp = 1 # prevent divide by 0
        wave.data = wave.data / Vpp * scale
        if wave.std is not None:
            ## TODO: since we don't have all the individual spike waveforms that what went
            ## into calculating wave.data, and since std is a non-linear operation,
            ## there's no easy way to update wave.std, so displayed error bars will not
            ## take normalization into account and may be larger than they should be
            pass
        return wave

    def draw(self):
        """Draw fill and LC to axes buffer (or whatever), avoiding unnecessary
        draws of all other artists in axes"""
        if self.fill != None:
            self.fill.draw()
        self.panel.ax.draw_artist(self.lc)


class Fill(object):
    """Fill slot, holds a PolyCollection of filled errors of visible neuron
    mean waveforms - only applicable in a SortPanel"""
    def __init__(self, panel, visible=False):
        self.panel = panel # panel that self belongs to
        self.pc = PolyCollection([],
                                 zorder=ERRORZORDER,
                                 alpha=ERRORALPHA,
                                 antialiased=True,
                                 visible=visible)
        self.panel.ax.add_collection(self.pc) # add to panel's axes' pool of PCs
        self.nverts = 0

    def update(self, waves, trefs):
        """Update PolyCollection vertex data from wave. Each entry in waves is a 2D array
        representing a multichannel spike. It's up to the caller to update colors if needed"""
        panel = self.panel
        allverts = []
        for wave, tref in zip(waves, trefs):
            nchans, nt = wave.std.shape
            # each chan has 2*nt number of points, nt +ve and nt -ve vertices;
            # for each vertex has two values, the x vals in col 0, yvals in col 1:
            verts = np.zeros((nchans, 2*nt, 2))
            data = panel.AD2uV(wave.data) # convert AD wave data to uV
            err = 1 * panel.AD2uV(wave.std) # convert AD wave std to uV, plot +/- 1 z-score
            if wave.ts is not None: # or maybe check if data.size != 0 too
                x = np.tile(wave.ts-tref, nchans) # nt*nchans 1D array
                x.shape = nchans, nt
                lower = self.panel.gain * (data - err)
                upper = self.panel.gain * (data + err)
                for chani, chan in enumerate(wave.chans):
                    vert = poly_between(x[chani], lower[chani], upper[chani])
                    vert = np.asarray(vert).T
                    # add offsets:
                    xpos, ypos = self.panel.pos[chan]
                    vert[:, 0] += xpos
                    vert[:, 1] += ypos
                    verts[chani] = vert
            allverts.append(verts)
        # flatten into a list of nt x 2 arrays:
        verts = [ row for vert in allverts for row in vert ]
        self.pc.set_verts(verts)
        self.nverts = len(verts)

    def show(self, enable=True):
        """Show/hide PC"""
        self.pc.set_visible(enable)

    def hide(self):
        """Hide PC"""
        self.show(False)

    def visible(self):
        """Visibility status"""
        return self.pc.get_visible()

    def set_colors(self, colors):
        """Set colors for PC"""
        if len(colors) != self.nverts:
            raise ValueError("Expected %d color values, one for each vert" % self.nverts)
        self.pc.set_facecolors(colors)
        #self.pc.set_edgecolors(colors) # not necessary, creates too much clutter

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
        colors = np.zeros(nsegments, dtype='|U7') # length-7 unicode strings
        segmenti = 0
        for spike in spikes:
            nchans = spike['nlockchans']
            # color segments according to each spike's max chan:
            colors[segmenti:segmenti+nchans] = self.panel.vcolors[spike['chan']]
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
        self.lc.set_color(list(colors))

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


class Stims(object):
    """Holds a LineCollection of stimulus onset/offset markers, one vertical line each"""
    def __init__(self, panel):
        self.panel = panel # panel that self belongs to
        self.lc = LineCollection([], linewidth=STIMSLINEWIDTH, linestyle=STIMSLINESTYLE,
                                 zorder=STIMSZORDER,
                                 antialiased=True,
                                 visible=True,
                                 pickradius=PICKRADIUS)
        self.panel.ax.add_collection(self.lc) # add to panel's axes' pool of LCs

    def update(self, stimtons, stimtoffs, tref):
        """Update LineCollection from stimtons and stimtoffs"""
        nrise, nfall = len(stimtons), len(stimtoffs)
        nsegments = nrise + nfall
        # 2 points per stim line, x vals in col 0, yvals in col 1
        segments = np.zeros((nsegments, 2, 2))
        colors = np.zeros(nsegments, dtype='|U7') # length-7 unicode strings
        xpos = 0
        ypos = np.array(list(self.panel.pos.values()))[:, 1]
        ylim = -ypos.max(), 2*ypos.max() # make sure it exceeds vertical limits of window
        segmenti = 0
        colors[:nrise] = GREEN # rising edge color
        colors[nrise:] = RED # falling edge color
        for stimtedge in np.concatenate([stimtons, stimtoffs]):
            x = stimtedge - tref + xpos
            segments[segmenti, :, 0] = x, x
            segments[segmenti, :, 1] = ylim
            segmenti += 1
        self.lc.set_segments(segments)
        self.lc.set_color(colors)

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
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

        self.spykewindow = parent.parent()

        self.plt = None # Plot object
        self.rasters = None # Rasters object
        self.stims = None # Stims object

        self.tw = tw # temporal window of each channel, in plot units (us ostensibly)
        self.cw = cw # temporal window of caret, in plot units

        self.figure.set_facecolor(BACKGROUNDCLR)
        # should really just turn off the edge line altogether, but how?
        self.figure.set_edgecolor(BACKGROUNDCLR)
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
        self.chans = probe.chans # sorted array of chans
        self.nchans = probe.nchans
        self.chans_selected = [] # for clustering, or potentially other uses as well

        # for mpl, convert probe SiteLoc to center bottom origin instead of center top
        siteloc = copy(self.SiteLoc) # lowercase means bottom origin
        ys = [ y for x, y in siteloc.values() ]
        maxy = max(ys)
        for key, (x, y) in siteloc.items():
            y = maxy - y
            siteloc[key] = (x, y) # update
        self.siteloc = siteloc # center bottom origin

        self.init_axes()
        self.pos = {} # positions of line centers, in plot units (us, uV)
        self.vcolors = {} # color mapping that cycles vertically in space
        self.do_layout() # defined by subclasses, sets self.pos
        self.xy_um = self.get_xy_um()
        x = self.xy_um[0]
        self.colxs = np.unique(x) # unique x values that demarcate columns
        self.colxs.sort() # guarantee they're in order from left to right
        self.ax.set_axis_off() # turn off the x and y axis
        self.ax.set_visible(True)
        self.ax.set_autoscale_on(False) # TODO: not sure if this is necessary
        self.draw()

        # add reference lines and caret in layered order
        self._show_tref(True) # call the _ methods directly, to prevent unnecessary draws
        self._show_vref(True)
        self._show_scale(True)
        self._show_caret(True)
        for ref in ['TimeRef', 'VoltageRef', 'Scale', 'Caret']:
            # enforce menu item toggle state
            self.spykewindow.__dict__['action%s' % ref].setChecked(True)
        self.draw() # do a full draw of the ref lines
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # init
        self.init_plots()
        self.init_rasters()
        self.init_stims()

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
                                       facecolor=BACKGROUNDCLR,
                                       frameon=False,
                                       alpha=1.0)

    def init_plots(self):
        """Create Plots for this panel"""
        chans = self.spykewindow.chans_enabled
        self.plt = Plot(chans, self, visible=True) # just one for this base class

    def init_rasters(self):
        """Init Rasters object"""
        self.rasters = Rasters(self)

    def init_stims(self):
        """Init Stims object"""
        self.stims = Stims(self)

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
            raise ValueError('Invalid ref: %r' % ref)

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
            raise ValueError('Invalid ref: %r' % ref)
        self.draw_refs()

    def draw_refs(self):
        """Redraw all enabled reflines, resave reflines_background"""
        plt = self.plt
        if plt != None: # has been initialized
            visible = plt.visible()
            plt.hide()
        self.show_rasters(False)
        self.show_stims(False)
        self.draw() # only draw all enabled refs - defined in FigureCanvas
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # update
        if plt != None: # has been initialized
            plt.show(visible) # re-show just the plot if it was previously visible
            plt.draw()
        self.show_rasters(True)
        self.show_stims(True)
        self.blit(self.ax.bbox)

    def _add_tref(self):
        """Add vertical time reference LineCollection, one line per probe column"""
        self.tlc = LineCollection([], linewidth=TREFLINEWIDTH,
                                  colors=TREFCLR,
                                  zorder=TREFLINEZORDER,
                                  antialiased=TREFANTIALIASED,
                                  visible=False)
        self.ax.add_collection(self.tlc) # add to axes' pool of LCs
        self._update_tref()

    def _add_vref(self):
        """Add horizontal voltage reference LineCollection, one line per probe channel"""
        self.vlc = LineCollection([], linewidth=VREFLINEWIDTH,
                                  colors=VREFCLR,
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
                                    colors=SCALECLR,
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
                               facecolor=CARETCLR,
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
        """Reset chans for this plot panel, triggering color update"""
        self.plt.chans = chans
        self.plt.update_colors()

    def get_xy_um(self):
        """Pull xy tuples in um out of self.pos, store in (2 x nchans) array,
        in self.chans order. In chart and lfp panels, this is different from siteloc,
        since these panels have only a single column"""
        xy_um = np.asarray([ (self.us2um(self.pos[chan][0]), self.uv2um(self.pos[chan][1]))
                                  for chan in self.chans ]).T # x is row0, y is row1
        return xy_um

    def get_closestchans(self, evt, n=1):
        """Return n closest channels to mouse event coords"""
        xdata = self.us2um(evt.xdata) # convert mouse event to um
        ydata = self.uv2um(evt.ydata)
        x, y = self.xy_um

        # minimize Euclidean distance:
        d2 = (x-xdata)**2 + (y-ydata)**2
        i = d2.argsort()[:n] # n indices sorted from smallest squared distance to largest
        chans = self.chans[i] # index into channels

        # Alternate strategy: Return n channels in column closest to mouse event coords,
        # sorted by vertical distance from mouse event:
        # what column is this event closest to? pick that column,
        # and then the n vertically closest chans within it
        '''
        # find nearest column
        dx = np.abs(xdata - self.colxs) # array of x distances
        coli = dx.argmin() # index of column nearest to mouse click
        colx = self.colxs[coli] # x coord of nearest column
        # indices into self.chans of chans that are in the nearest col:
        i, = (x == colx).nonzero()
        colchans = np.asarray(self.chans)[i] # channels in nearest col
        # vertical distances between mouse click and all chans in this col:
        dy = np.abs(y[i] - ydata)
        i = dy.argsort()[:n] # n indices sorted from smallest to largest y distance
        chans = colchans[i] # index into channels in the nearest column
        '''

        if len(chans) == 1:
            chans = chans[0] # pull it out, return a single value
        return chans

    def plot(self, wave, tref=None):
        """Plot waveforms and optionally rasters and stim times wrt a reference time point"""
        if tref == None:
            tref = wave.ts[0] # use first timestamp in waveform as the reference time point
        self.restore_region(self.reflines_background)
        # update plots and rasters:
        self.plt.update(wave, tref)
        self.plt.draw()
        if self.spykewindow.actionRasters.isChecked() and self.rasters != None:
            self.update_rasters(tref)
            self.rasters.draw()
        if self.spykewindow.actionStims.isChecked() and self.stims != None:
            self.update_stims(tref)
            self.stims.draw()
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

    def update_stims(self, tref):
        """Update stimulus rise and fall positions and visibility wrt tref"""
        # find out which stim edges are within time window:
        spw = self.spykewindow
        lo, hi = spw.stimtons.searchsorted((tref+self.tw[0], tref+self.tw[1]))
        stimtons = spw.stimtons[lo:hi] # stim onsets within range of current time window
        lo, hi = spw.stimtoffs.searchsorted((tref+self.tw[0], tref+self.tw[1]))
        stimtoffs = spw.stimtoffs[lo:hi] # stim offsets within range of current time window
        self.stims.update(stimtons, stimtoffs, tref)

    def show_rasters(self, enable=True):
        """Show/hide all rasters in this panel"""
        if self.rasters != None:
            self.rasters.show(enable)

    def show_stims(self, enable=True):
        """Show/hide all stimulus edges in this panel"""
        if self.stims != None:
            self.stims.show(enable)

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
        self.spykewindow.seek(self.plt.tref + direction*self.stream.tres)
        evt.Skip() # allow left, right, pgup and pgdn to propagate to OnKeyDown handler
    '''
    def OnButtonPress(self, evt):
        """Seek to timepoint as represented on chan closest to left click.
        Toggle specific chans on right click. On ctrl+left click, reset primary
        peak timepoint and maxchan of currently selected spike. On ctrl+right click,
        reset secondary peak timepoint"""
        spw = self.spykewindow
        button = evt.button
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
        # NOTE: evt.key is supposed to give us the modifier, if any (like ctrl or shift)
        # but doesn't seem to work in MPL in qt. Also, evt.guiEvent always seems to be
        # None in qt. Also, up and down scroll events don't work.
        chan = self.get_closestchans(evt, n=1)
        # find clicked timepoint:
        xpos = self.pos[chan][0]
        # undo position correction and convert from relative to absolute time:
        t = evt.xdata - xpos + self.plt.tref
        t = spw.get_nearest_timepoint(t)
        if ctrl:
            if button == 1: # left click
                # set t as primary peak and align selected spike to it, set maxchan
                self.alignselectedspike('primary', t, chan)
                spw.seek(t) # seek to t
            elif button == 3: # right click
                # designate t as secondary peak of selected spike
                self.alignselectedspike('secondary', t)
                spw.seek(t) # seek to t
        else:
            if button == 1: # left click
                spw.seek(t) # seek to t
            elif button == 3: # right click
                if spw.has_sort:
                    print("Channel toggling is disabled when sort is open")
                else:
                    # toggle closest chan, but only when there's no sort:
                    if chan not in spw.chans_enabled: # enable chan
                        spw.chans_enabled = np.union1d(spw.chans_enabled, [chan])
                    else: # disable chan
                        spw.chans_enabled = np.setdiff1d(spw.chans_enabled, [chan])

    def alignselectedspike(self, peaktype, t, chan=None):
        """Align spike selected in sortwin to t, where t is designated as the
        primary or secondary peak timepoint. Also optionally set the maxchan
        of the spike to chan. Since this is happening in a DataWindow, it's safe
        to assume that a .srf or .track file is open"""
        #if srff not open:
        #    print("can't align selected spike without .srf file(s)")
        spw = self.window().parent() # spyke window
        spikes = spw.sort.spikes
        try:
            sid = spw.GetSpike()
        except RuntimeError as err:
            print(err)
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
        except RuntimeError as err:
            print(err)
            return
        abort = False
        try:
            if spw.primarypeakt == None or spw.secondarypeakt == None:
                abort = True
        except AttributeError:
            abort = True
        if abort:
            print("New primary and secondary peaks need to be set before "
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

        # mark sid as dirty in .wave file:
        spw.update_dirtysids([sid])

        # reset for next alignment session:
        spw.primarypeakt = None
        spw.secondarypeakt = None
        spw.alignspike2chan = None

        # reload spike in sort panel:
        sortwin = spw.windows['Sort']
        sortwin.panel.update_plots()

        # seek to new timepoint, this also automatically updates the raster line:
        spw.seek(t)
        print('Realigned and reloaded spike %d to t=%d on chan %d' % (sid, t, chan))

    '''
    def OnPick(self, evt):
        """Pop up a tooltip when mouse is within PICKTHRESH of a line"""
        tooltip = self.GetToolTip()
        if evt.mouseevent.inaxes:
            # assume it's one of our SpykeLines, since those are the only ones with their
            # .picker attrib enabled:
            line = evt.artist
            chan = line.chan
            xpos, ypos = self.pos[chan]
            # undo position correction and convert from relative to absolute time:
            t = evt.mouseevent.xdata - xpos + self.plt.tref
            v = (evt.mouseevent.ydata - ypos) / self.gain
            if t >= self.stream.t0 and t <= self.stream.t1: # in bounds
                # round to nearest (possibly interpolated) sample:
                t = intround(t / self.stream.tres) * self.stream.tres
                tip = 'ch%d\n' % chan + \
                      't=%d %s\n' % (t, 'us') + \
                      'V=%.1f %s\n' % (v, 'uV') + \
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
        if not sortpanel and (chan not in self.plt.chans):
            self.setToolTip('')
            return
        xpos, ypos = self.pos[chan]
        t = evt.xdata - xpos
        if not sortpanel:
            t += self.plt.tref
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
        t = intround(t / tres) * tres # nearest sample
        tip = 'ch%d @ %r %s\n' % (chan, self.SiteLoc[chan], 'um') + \
              't=%.1f %s\n' % (t, 'us') + \
              'V=%.1f %s\n' % (v, 'uV') + \
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
        # event.angleDelta() is a multiple of 120:
        # https://doc-snapshots.qt.io/qt5-dev/qwheelevent.html#angleDelta
        di = event.angleDelta().y() / 120
        sign = np.sign(di)
        absdi = abs(di)
        if ctrl: # scale voltage
            if sign == 1:
                self.gain = self.gain * (1 + absdi / SCALEX)
            else:
                self.gain = self.gain / (1 + absdi / SCALEX)
            print('%s window gain=%g' % (wintype, self.gain))
        elif shift: # scale time
            if sign == 1:
                tw = tuple([t / (1 + absdi / SCALEX) for t in self.tw])
            else:
                tw = tuple([t * (1 + absdi / SCALEX) for t in self.tw])
            self.update_tw(tw) # update Panel display tw
            spw.__dict__[wintype2wintw[wintype]] = tw # update spyke window's data fetch tw
            print('%s window tw=(%g, %g) ms' % (wintype, self.tw[0]/1000, self.tw[1]/1000))
        else: # step left/right on wheel up/down
            win = self.parent()
            win.step(-di)
        spw.plot(wintypes=[wintype])


class SpikePanel(PlotPanel):
    """Spike panel. Presents a narrow temporal window of all channels
    laid out according to self.siteloc"""
    RASTERHEIGHT = 75 # uV, TODO: calculate this instead

    def __init__(self, *args, **kwargs):
        self.gain = 1.5
        PlotPanel.__init__(self, *args, **kwargs)

    def init_stims(self):
        """Disable for SpikePanel"""
        pass

    def do_layout(self):
        # chans ordered bottom to top, then left to right:
        hchans = self.probe.chans_btlr
        # chans ordered left to right, then bottom to top
        vchans = self.probe.chans_lrbt
        #print('Horizontal ordered chans in Spikepanel:\n%r' % hchans)
        # x origin is somewhere in between the xlimits. xlimits are asymmetric
        # if self.tw is asymmetric:
        self.ax.set_xlim(self.um2us(self.siteloc[hchans[0]][0]) + self.tw[0],
                         self.um2us(self.siteloc[hchans[-1]][0]) + self.tw[1])
        self.ax.set_ylim(self.um2uv(self.siteloc[vchans[0]][1]) - CHANVBORDER,
                         self.um2uv(self.siteloc[vchans[-1]][1]) + CHANVBORDER)
        colorgen = itertools.cycle(iter(PLOTCLRS))
        for chan in vchans:
            # chan order doesn't matter for setting .pos, but it does for setting .colors
            self.pos[chan] = (self.um2us(self.siteloc[chan][0]),
                              self.um2uv(self.siteloc[chan][1]))
            # assign colors so that they cycle vertically in space:
            self.vcolors[chan] = next(colorgen)

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
        # chans ordered left to right, then bottom to top:
        vchans = self.probe.chans_lrbt
        self.ax.set_xlim(0 + self.tw[0], 0 + self.tw[1]) # x origin at center
        miny = self.um2uv(self.siteloc[vchans[0]][1])
        maxy = self.um2uv(self.siteloc[vchans[-1]][1])
        # average vertical spacing between chans, in uV:
        ngaps = max(self.nchans-1, 1) # at least 1
        vspace = (maxy - miny) / ngaps
        self.ax.set_ylim(miny - CHANVBORDER, maxy + CHANVBORDER)
        colorgen = itertools.cycle(iter(PLOTCLRS))
        for chani, chan in enumerate(vchans):
            #self.pos[chan] = (0, self.um2uv(self.siteloc[chan][1])) # x=0 centers horizontally
            # x=0 centers horizontally, equal vertical spacing:
            self.pos[chan] = (0, chani*vspace)
            # assign colors so that they cycle vertically in space:
            self.vcolors[chan] = next(colorgen)

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
        # get it right for first plt.update() call:
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
        try: self.stream.layout
        except AttributeError: return
        newchans, newpos = [], {}
        for chan in self.stream.layout.chans:
            newchans.append(chan)
            newpos[chan] = self.pos[chan]
        self.chans = np.asarray(newchans)
        self.pos = newpos

    def set_chans(self, chans):
        """Reset chans for this LFPPanel, triggering color update. Take intersection of
        stream file's channels and chans, conserving order in stream file's channels.
        This overloads ChartPanel.set_chans only to handle the special case of .srf LFP"""
        stream = self.stream
        if stream.ext == '.srf': # single or MultiStream .srf
            streamfilechans = stream.layout.chans # SurfStream should have a layout attrib
            chans = [ chan for chan in streamfilechans if chan in chans ]
        ChartPanel.set_chans(self, chans)

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
        self.manual_chan_selection = False
        self.enable_fills = False # global enable flag for all fills
        self.sortwin = self.parent()
        self.sids = []
        self.nids = []

    def get_AD2uV(self):
        try: # use sort by default:
            return self.sort.converter.AD2uV
        except AttributeError: # sort doesn't exist yet
            return self.stream.converter.AD2uV

    AD2uV = property(get_AD2uV) # convenience for Plot objects to reference

    def get_tres(self):
        return self.sort.tres # override PlotPanel's definition

    tres = property(get_tres)

    def init_plots(self):
        """Init one SpikePlot for neurons and one for spikes"""
        self.splt = SpikePlot(self, linewidth=SPIKELINEWIDTH, linestyle=SPIKELINESTYLE,
                              alpha=0.5)
        self.nplt = SpikePlot(self, linewidth=NEURONLINEWIDTH, linestyle=NEURONLINESTYLE,
                              alpha=1.0)
        self.nplt.fill = Fill(self) # to show variance of member spikes

    def init_rasters(self):
        """Disable for SortPanel"""
        pass

    def init_stims(self):
        """Disable for SortPanel"""
        pass

    def update_tw(self, tw):
        """Same as parent, but auto-refresh all plots after"""
        PlotPanel.update_tw(self, tw)
        self.update_plots()

    def draw_refs(self):
        """Redraw all enabled reflines, resave reflines_background"""
        plots = [self.splt, self.nplt, self.nplt.fill]
        visibles = [False, False, False]
        for i, plt in enumerate(plots):
            if plt != None: # has been initialized
                visibles[i] = plt.visible()
                plt.hide()
        self.draw() # only draw all enabled refs - defined in FigureCanvas
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # update
        for plt, viz in zip(plots, visibles):
            if plt != None: # has been initialized
                plt.show(viz) # re-show the plot if it was previously visible
                plt.draw()
        self.blit(self.ax.bbox)

    def _add_vref(self):
        """Increase pick radius for vrefs from default zero, since we're
        relying on them for tooltips"""
        PlotPanel._add_vref(self)
        self.vlc.set_pickradius(PICKRADIUS)

    def plot_spikes(self, sids):
        """Replace any currently plotted spikes with sids"""
        colors, waves, trefs = [], [], []
        for sid in sids:
            t = self.sort.spikes['t'][sid]
            nid = self.sort.spikes['nid'][sid]
            wave = self.sort.get_wave(sid)
            '''
            # lower level method:
            nchans = s.spikes['nchans'][sid]
            chans = s.spikes['chans'][sid, :nchans]
            wavedata = self.wavedata[sid, 0:nchans]
            t0 = spikes['t0'][sid]
            t1 = spikes['t1'][sid]
            '''
            # slice wave according to time window of this panel:
            wave = wave[t+self.tw[0] : t+self.tw[1]]
            waves.append(wave)
            trefs.append(t)
            nchans = len(wave.chans)
            clrs = [CLUSTERCLRDICT[nid]] * nchans # collect one color value per chan
            colors.extend(clrs)
        self.splt.update(waves, trefs)
        self.splt.set_colors(colors)
        self.splt.show()
        self.sids = sids # save for update_plots() call

    def plot_neurons(self, nids):
        """Replace any currently plotted neurons with nids"""
        colors, waves, trefs = [], [], []
        for nid in nids:
            n = self.sort.neurons[nid]
            t = n.t
            wave = n.get_wave() # calls n.update_wave() if necessary
            # slice wave according to time window of this panel:
            wave = wave[t+self.tw[0] : t+self.tw[1]]
            waves.append(wave)
            trefs.append(t)
            nchans = len(wave.chans)
            clrs = [CLUSTERCLRDICT[nid]] * nchans # collect one color value per chan
            colors.extend(clrs)
        self.nplt.update(waves, trefs)
        self.nplt.set_colors(colors)
        self.nplt.show()
        self.nids = nids # save for update_plots() call

    def clear_draw_blit(self):
        """Clear all previous plots, draw spikes and neurons, blit to screen"""
        self.restore_region(self.reflines_background)
        self.splt.draw()
        self.nplt.draw()
        self.blit(self.ax.bbox)

    def clear_spikes(self):
        """Clear display of all spikes"""
        self.plot_spikes([])
        self.clear_draw_blit()

    def clear_plots(self):
        """Clear display of all spikes and neurons"""
        self.plot_spikes([])
        self.plot_neurons([])
        self.clear_draw_blit()

    def show_fills(self, enable=True):
        """Toggle visibility of all currently bound fills"""
        self.enable_fills = enable # update global flag
        self.nplt.fill.show(enable)
        self.clear_draw_blit()

    def update_plots(self):
        """Update display of all spikes and neurons"""
        self.plot_spikes(self.sids)
        self.plot_neurons(self.nids)
        self.clear_draw_blit()

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
            chan = int(self.get_closestchans(evt, n=1)) # from int64 for clean jsonpickle
            if chan not in self.chans_selected: # it's unselected, select it
                self.chans_selected.append(chan)
            else: # it's selected, unselect it
                self.chans_selected.remove(chan)
            self.manual_chan_selection = True
        elif button == 2: # middle click
            self.sortwin.spykewindow.plotButton.click() # same as hitting ENTER in nslist
            #self.sortwin.on_actionSelectRandomSpikes_triggered()
        elif button == 3: # right click
            self.chans_selected = [] # clear channel selection
            self.manual_chan_selection = False
        self.update_selvrefs()
        self.draw_refs() # update

    def update_selvrefs(self):
        """Set line widths, lengths, and colors of vrefs according to chans in
        self.chans_selected. Note that any changes to the display code here should also be
        made in the timepoint selection code for dimension reduction in
        SortWindow.get_tis()"""
        segments = self.segments.copy()
        colors = np.repeat(VREFCLR, len(self.pos)) # clear all lines
        linewidths = np.repeat(VREFLINEWIDTH, len(self.pos)) # clear all lines
        inclt = self.sortwin.inclt
        # scale time selection around t=0 according to window asymmetry:
        dtw = self.sort.tw[1] - self.sort.tw[0] # spike time window width
        incltleft = intround(abs(self.tw[0]) / dtw * inclt) # left fraction wrt xpos
        incltright = inclt - incltleft # right fraction wrt xpos
        #print("self.tw, incltleft, incltright", self.tw, incltleft, incltright)
        for chan in self.chans_selected: # set line color of selected chans
            vrefsegmenti = self.chan2vrefsegmenti[chan] # one vref for every enabled chan
            xpos = self.pos[chan][0] # chan xpos center (us)
            # modify the x values of this segment:
            x0, x1 = intround(xpos-incltleft), intround(xpos+incltright)
            segments[vrefsegmenti][:, 0] = x0, x1
            colors[vrefsegmenti] = VREFSELECTEDCLR
            linewidths[vrefsegmenti] = SELECTEDVREFLINEWIDTH
        self.vlc.set_segments(segments)
        self.vlc.set_color(list(colors))
        self.vlc.set_linewidth(list(linewidths))


class SpikeSortPanel(SortPanel, SpikePanel):
    def __init__(self, parent, tw=None):
        self.gain = 1.0
        SortPanel.__init__(self, parent, tw=tw)

    def wheelEvent(self, event):
        """Scroll gainComboBox or incltComboBox on mouse wheel scroll"""
        modifiers = event.modifiers()
        ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
        sw = self.window() # SortWindow
        if ctrl: # scroll gainComboBox
            cbox = sw.gainComboBox
            on_box_triggered = sw.on_gainComboBox_triggered
        else: # scroll incltComboBox
            cbox = sw.incltComboBox
            on_box_triggered = sw.on_incltComboBox_triggered
        nitems = cbox.count()
        di = intround(event.angleDelta().y() / 120)
        # both combo boxes are sorted in decreasing order, hence the negation of di:
        i = min(max(cbox.currentIndex()-di, 0), nitems-1)
        cbox.setCurrentIndex(i)
        on_box_triggered() # as if it were user selected
        event.accept() # ensure event doesn't propagate to parent, prevent view scrolling

#class ChartSortPanel(SortPanel, ChartPanel):
