"""Core classes and functions used throughout spyke"""

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

__authors__ = ['Martin Spacek', 'Reza Lotun']

import hashlib
import time
import datetime
import os

import random
import string
from copy import copy

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
getSaveFileName = QtGui.QFileDialog.getSaveFileName

import numpy as np
from numpy import pi
import scipy.signal

import matplotlib as mpl

# set some numpy options - these should hold for all modules in spyke
np.set_printoptions(precision=3)
np.set_printoptions(threshold=1000)
np.set_printoptions(edgeitems=5)
np.set_printoptions(linewidth=150)
np.set_printoptions(suppress=True)
# make overflow, underflow, div by zero, and invalid all raise errors
# this really should be the default in numpy...
np.seterr(all='raise')

UNIXEPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0) # UNIX epoch: Jan 1, 1970

NULL = '\x00'

MU = '\xb5' # greek mu symbol
MICRO = 'u'

DEFDATFILTMETH = 'BW' # default .dat filter method: None, 'BW', 'WMLDR'
DEFNSXFILTMETH = 'BW' # default .nsx filter method: None, 'BW', 'WMLDR'
BWHPF0 = 300 # butterworth high-pass filter low-frequency cutoff, Hz
BWLPF1 = 300 # butterworth low-pass filter high-frequency cutoff, Hz
BWHPORDER = 4 # butterworth high-pass filter order
BWLPORDER = 4 # butterworth low-pass filter order
# don't filter raw data, only decimate to get low pass? simple, but aliases
LOWPASSFILTER = False

DEFCAR = 'Median' # default common average reference method: None, 'Median', 'Mean';
                  # 'Median' works best because it's least affected by spikes

DEFHPRESAMPLEX = 2 # default highpass resampling factor for all stream types
DEFLPSAMPLFREQ = 1000 # default lowpass sampling rate for wide-band stream types, Hz
DEFHPDATSHCORRECT = False ## TODO: this may not hold for open-ephys and Intan chips!
DEFHPNSXSHCORRECT = False # no need for .nsx files, s+h delay is only 1 ns between chans
DEFHPSRFSHCORRECT = True

# Apparently KERNELSIZE == number of kernel zero crossings, but that seems to depend on
# the phase of the kernel, some have one less, depending on the channel (when doing sample
# and hold correction). Anyway, total number of points in the kernel is KERNELSIZE plus 1
# (for the middle point) - see Blanche2006.
# Should kernel size depend on the sampling rate? No, but perhaps the minimum kernel
# size depends on the Nyquist rate.
KERNELSIZE = 12
# kernel size needs to be even, otherwise there's a slight but undesireable time shift,
# perhaps because sampfreq always needs to be an integer multiple of rawsampfreq:
assert KERNELSIZE % 2 == 0
# number of excess raw datapoints to include on either side of each wideband Stream
# (such as a DATStream or NSXStream) during a slice call. Due to the lack of analog filtering,
# a greater excess is needed than e.g. SurfStream because it's already analog filtered
SRFNCHANSPERBOARD = 32 # TODO: would be better to not hard-code this
XSWIDEBANDPOINTS = 200

MAXLONGLONG = 2**63-1
MAXNBYTESTOFILE = 2**31 # max array size safe to call .tofile() on in Numpy 1.5.0 on Windows

MAXNSPIKEPLOTS = 200
MAXNROWSLISTSELECTION = 10000

CHANFIELDLEN = 256 # channel string field length at start of .resample file

INVPI = 1 / pi


class EmptyClass(object):
    pass


class Converter(object):
    """Store intgain and extgain values and provide methods to convert between AD and uV
    values for .srf files, even when a Stream (where intgain and extgain are stored) isn't
    available"""
    def __init__(self, intgain, extgain):
        self.intgain = intgain
        self.extgain = extgain

    def AD2uV(self, AD):
        """Convert rescaled AD values to float32 uV
        Biggest +ve voltage is 10 million uV, biggest +ve rescaled signed int16 AD val
        is half of 16 bits, then divide by internal and external gains

        TODO: unsure: does the DT3010 acquire from -10 to 10 V at intgain == 1 and encode
        that from 0 to 4095?
        """
        return np.float32(AD) * 10000000 / (2**15 * self.intgain * self.extgain)

    def uV2AD(self, uV, dtype=np.int16):
        """Convert uV to signed rescaled AD values of type dtype"""
        return dtype(np.round(uV * (2**15 * self.intgain * self.extgain) / 10000000))


class Converter_TSF_1002(object):
    """Store intgain and extgain values and provide methods to convert between AD and uV
    values, even when a Stream (where intgain and extgain are stored) isn't available. Meant
    specifically for .tsf version 1002 files, which have no specific AD voltage limits, and
    already come as signed values centered around 0 V"""
    def __init__(self, intgain, extgain):
        self.intgain = intgain # uV per AD value
        self.extgain = extgain

    def AD2uV(self, AD):
        """Convert signed int16 AD values to float32 uV"""
        return np.float32(AD) * self.intgain * self.extgain

    def uV2AD(self, uV, dtype=np.int16):
        """Convert float32 uV to signed AD values of type dtype"""
        return dtype(np.round(uV / (self.intgain * self.extgain)))


class SimpleConverter(object):
    """Store conversion factors between AD values and uV values, and provide
    methods to convert between them, even when a stream isn't available. Note that
    conceptually, AD2uVx is identical to uVperAD"""
    def __init__(self, AD2uVx):
        self.AD2uVx = AD2uVx
        self.uV2ADx = 1 / AD2uVx

    def AD2uV(self, AD):
        return self.AD2uVx * np.float32(AD)
        
    def uV2AD(self, uV, dtype=np.int16):
        return dtype(np.round(self.uV2ADx * uV))


class DatConverter(SimpleConverter):
    pass


class NSXConverter(SimpleConverter):
    pass


class WaveForm(object):
    """Just a container for data, std of data, timestamps, and channels.
    Sliceable in time, and indexable in channel space. Only really used for
    convenient plotting. Everything else uses the sort.wavedata array, and
    related sort.spikes fields"""
    def __init__(self, data=None, std=None, ts=None, chans=None):
        self.data = data # in AD, potentially multichannel, depending on shape
        self.std = std # std of data
        self.ts = ts # timestamps array in us, one for each sample (column) in data
        self.chans = chans # channel ids corresponding to rows in .data

    def __getitem__(self, key):
        """Make waveform data sliceable in time, and directly indexable by channel id(s).
        Return a new WaveForm"""
        
        # check for std field, won't exist for old saved Waveforms in .sort files:
        try: self.std
        except AttributeError: self.std = None
        
        if type(key) == slice: # slice self in time
            if self.ts is None:
                return WaveForm() # empty WaveForm
            else:
                lo, hi = self.ts.searchsorted([key.start, key.stop])
                data = self.data[:, lo:hi]
                if self.std is None:
                    std = None
                else:
                    std = self.std[:, lo:hi]
                ts = self.ts[lo:hi]
                '''
                if np.asarray(data == self.data).all() and np.asarray(ts == self.ts).all():
                    # no need for a new WaveForm, though new WaveForms aren't expensive,
                    # only new data are
                    return self
                '''
                # return a new WaveForm:
                return WaveForm(data=data, std=std, ts=ts, chans=self.chans)
        else: # index into self by channel id(s)
            keys = toiter(key)
            #try: assert (self.chans == np.sort(self.chans)).all() # testing code
            #except AssertionError: import pdb; pdb.set_trace() # testing code
            try:
                assert set(keys).issubset(self.chans), ("requested channels outside of "
                                                        "channels in waveform")
                # this is fine:
                #assert len(set(keys)) == len(keys), "same channel specified more than once"
            except AssertionError:
                raise IndexError('invalid index %r' % key)
            #i = self.chans.searchsorted(keys) # indices into rows of data
            # best not to assume that chans are sorted, often the case in LFP data;
            # i are indices into rows of data:
            i = [ int(np.where(chan == self.chans)[0]) for chan in keys ]
            data = self.data[i] # grab the appropriate rows of data
            if self.std is None:
                std = None
            else:
                std = self.std[i]
            return WaveForm(data=data, std=std, ts=self.ts, chans=keys) # return a new WaveForm

    def __len__(self):
        """Number of data points in time"""
        nt = len(self.ts)
        assert nt == self.data.shape[1] # obsessive
        return nt

    def _check_add_sub(self, other):
        """Check a few things before adding or subtracting waveforms"""
        if self.data.shape != other.data.shape:
            raise ValueError("Waveform shapes %r and %r don't match" %
                             (self.data.shape, other.data.shape))
        if self.chans != other.chans:
            raise ValueError("Waveform channel ids %r and %r don't match" %
                             (self.chans, other.chans))

    def __add__(self, other):
        """Return new waveform which is self+other. Keep self's timestamps"""
        self._check_add_sub(other)
        return WaveForm(data=self.data+other.data,
                        ts=self.ts, chans=self.chans)

    def __sub__(self, other):
        """Return new waveform which is self-other. Keep self's timestamps"""
        self._check_add_sub(other)
        return WaveForm(data=self.data-other.data,
                        ts=self.ts, chans=self.chans)
    '''
    def get_padded_data(self, chans):
        """Return self.data corresponding to self.chans,
        padded with zeros for chans that don't exist in self"""
        common = set(self.chans).intersection(chans) # overlapping chans
        dtype = self.data.dtype # self.data corresponds to self.chans
        # padded_data corresponds to chans:
        padded_data = np.zeros((len(chans), len(self.ts)), dtype=dtype)
        chanis = [] # indices into self.chans corresponding to overlapping chans
        commonis = [] # indices into chans corresponding to overlapping chans
        for chan in common:
            chani, = np.where(chan == np.asarray(self.chans))
            commoni, = np.where(chan == np.asarray(chans))
            chanis.append(chani)
            commonis.append(commoni)
        chanis = np.concatenate(chanis)
        commonis = np.concatenate(commonis)
        # for overlapping chans, overwrite the zeros with data:
        padded_data[commonis] = self.data[chanis]
        return padded_data
    '''
           
class SpykeToolWindow(QtGui.QMainWindow):
    """Base class for all of spyke's tool windows"""
    def __init__(self, parent, flags=Qt.Tool):
        QtGui.QMainWindow.__init__(self, parent, flags)
        self.maximized = False

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        shift = modifiers == Qt.ShiftModifier # only modifier is shift
        if key == Qt.Key_F11:
            self.toggleMaximized()
        elif key == Qt.Key_S and shift:
            self.on_actionSave_triggered()
        else:
            QtGui.QMainWindow.keyPressEvent(self, event) # pass it on

    def mouseDoubleClickEvent(self, event):
        """Doesn't catch window titlebar doubleclicks for some reason (window manager
        catches them?). Have to doubleclick on a part of the window with no widgets in it"""
        self.toggleMaximized()

    def closeEvent(self, event):
        # remove 'Window' from class name
        windowtype = type(self).__name__.replace('Window', '')
        self.parent().HideWindow(windowtype)

    def toggleMaximized(self):
        if not self.maximized:
            self.normalPos, self.normalSize = self.pos(), self.size()
            dw = QtGui.QDesktopWidget()
            rect = dw.availableGeometry(self)
            self.setGeometry(rect)
            self.maximized = True
        else: # restore
            self.resize(self.normalSize)
            self.move(self.normalPos)
            self.maximized = False

    def on_actionSave_triggered(self):
        """Save panel to file"""
        f = self.panel.figure

        # copied and adapted from mpl.backend_qt4.NavigationToolbar2QT.save_figure():
        filetypes = f.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        default_filetype = f.canvas.get_default_filetype()

        startpath = mpl.rcParams.get('savefig.directory', '')
        startpath = os.path.expanduser(startpath)
        start = os.path.join(startpath, f.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)
        fname = getSaveFileName(self.panel, "Save panel to",
                                start, filters, selectedFilter)
        if fname:
            fname = str(fname) # convert from QString
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                mpl.rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                mpl.rcParams['savefig.directory'] = os.path.dirname(str(fname))
            try:
                f.canvas.print_figure(fname, facecolor=None, edgecolor=None)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self.panel, "Error saving file", str(e),
                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)
            print('panel saved to %r' % fname)


class SpykeListView(QtGui.QListView):
    def __init__(self, parent):
        QtGui.QListView.__init__(self, parent)
        self.sortwin = parent
        #self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QtGui.QListView.ExtendedSelection)
        self.setLayoutMode(QtGui.QListView.Batched) # prevents lockup during huge layout ops
        # Setting resize mode to "adjust" sometimes results in a bug where Qt seems to
        # be reflowing the contents many times over before it finally stops, resulting in
        # very slow operations when changing list contents (like adding/removing neurons).
        # But, with this disabled, the contents no longer reflow, and you're forced to use
        # scrollbars unnecessarily to see all the list contents. This might also be
        # interacting with the setWrapping and/or setBatchSize features:
        #self.setResizeMode(QtGui.QListView.Adjust) # recalculates layout on resize
        self.setUniformItemSizes(True) # speeds up listview
        self.setFlow(QtGui.QListView.LeftToRight) # default is TopToBottom
        self.setWrapping(True)
        self.setBatchSize(300)
        #self.setViewMode(QtGui.QListView.IconMode)

    def mousePressEvent(self, event):
        sw = self.sortwin
        buttons = event.buttons()
        if buttons == QtCore.Qt.LeftButton:
            QtGui.QListView.mousePressEvent(self, event) # handle as usual
        else:
            self.sortwin.mousePressEvent(event) # pass on up to Sort window

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        ctrldown = bool(modifiers & Qt.ControlModifier)
        ctrlup = not ctrldown
        if (key in [Qt.Key_A, Qt.Key_M, Qt.Key_G, Qt.Key_Equal, Qt.Key_Minus, Qt.Key_Slash,
                    Qt.Key_P, Qt.Key_Backslash, Qt.Key_NumberSign, Qt.Key_F, Qt.Key_R,
                    Qt.Key_E, Qt.Key_B, Qt.Key_BracketLeft, Qt.Key_BracketRight,
                    Qt.Key_Comma, Qt.Key_Period, Qt.Key_C, Qt.Key_T, Qt.Key_W]
            or ctrlup and key == Qt.Key_Space):
            event.ignore() # pass it on up to the parent
        else:
            QtGui.QListView.keyPressEvent(self, event) # handle it as usual

    def selectionChanged(self, selected, deselected, prefix=None):
        """Plot neurons or spikes on list item selection"""
        # For short lists, display the actual selection in the list, otherwise, if there are
        # too many entries in the list, selection gets unbearably slow, especially as you
        # select items further down the list. So for very long lists, don't actually show the
        # selection. The selection events all still seem to work though, and for some reason
        # sometimes the selections themselves are displayed, even when selected
        # programmatically:
        if self.nrows < MAXNROWSLISTSELECTION:
            QtGui.QListView.selectionChanged(self, selected, deselected)
        panel = self.sortwin.panel
        addis = [ i.data().toInt()[0] for i in selected.indexes() ]
        remis = [ i.data().toInt()[0] for i in deselected.indexes() ]
        panel.removeItems([ prefix+str(i) for i in remis ])
        # for speed, don't allow more than MAXNSPIKEPLOTS spikes to be plotted in sort panel:
        if prefix == 's':
            '''
            # note that self.nrowsSelected seems to report nrows selected *including* those
            # added and removed by the current selection event
            net = len(addis) - len(remis)
            print('num selected %d' % self.nrowsSelected)
            print('net change is %d' % net)
            nwereselected = self.nrowsSelected - net
            print('num were selected is %d' % nwereselected)
            maxnadd = max(MAXNSPIKEPLOTS - nwereselected + len(remis), 0)
            print('maxnadd is %d' % maxnadd)
            addis = addis[:maxnadd]
            '''
            nadd = len(addis)
            maxnadd = max(MAXNSPIKEPLOTS - self.nrowsSelected + nadd, 0)
            if maxnadd == 0:
                return
            if nadd > maxnadd:
                # if we can't add all the requested spikes to the sort panel without
                # exceeding MAXNSPIKEPLOTS, then randomly sample however many we can still
                # add (maxnadd), and add them to the sort panel
                print('adding %d randomly sampled plots of %d selected spikes'
                      % (maxnadd, self.nrowsSelected))
                addis = random.sample(addis, maxnadd)
                panel.maxed_out = True
            else:
                panel.maxed_out = False
        #t0 = time.time()
        panel.addItems([ prefix+str(i) for i in addis ])
        #print('addItems took %.3f sec' % (time.time()-t0))
        #print("done selchanged, %r, addis=%r, remis=%r" % (prefix, addis, remis))

    def updateAll(self):
        self.model().updateAll()

    def get_nrows(self):
        return self.model().rowCount()

    nrows = property(get_nrows)

    def selectRows(self, rows, on=True, scrollTo=False):
        """Row selection in listview is complex. This makes it simpler"""
        ## TODO: There's a bug here, where if you select the last two neurons in nlist,
        ## (perhaps these last two need to be near a list edge), merge them, and then
        ## undo, then merge again (instead of just redoing), then undo again, they're
        ## both selected, but only the first is replotted because the selchanged event
        ## is only passed the first of the two as being newly selected. If however
        ## before remerging, you clear the selection or select something else, and then
        ## go back and select those same two neurons and merge, and undo, it works fine,
        ## and the selchanged event gets both items as newly selected. Seems like a Qt
        ## bug, or at least some very subtle timing problem of some kind. This might have
        ## something to do with reflow when changing list contents, but even resetting
        ## listview behaviour to default doesn't make this go away. Also, seems to happen
        ## for selection of one index at a time, and for doing it all in one go with a
        ## QItemSelection.
        
        rows = toiter(rows)
        m = self.model()
        sm = self.selectionModel()
        if on:
            flag = sm.Select
        else:
            flag = sm.Deselect
        #print('start select=%r loop for rows %r' % (on, rows))
        '''
        # unnecessarily emits nrows selectionChanged signals, causes slow
        # plotting in mpl commit 50fc548465b1525255bc2d9f66a6c7c95fd38a75 (pre
        # 1.0) and later:
        [ sm.select(m.index(row), flag) for row in rows ]
        '''
        # emits single selectionChanged signal, more efficient, but causes a bit of
        # flickering, or at least used to in Qt 4.7.0:
        sel = QtGui.QItemSelection()
        for row in rows:
            index = m.index(row)
            #print('row: %r, index: %r' % (row, index))
            sel.select(index, index) # topleft to bottomright
        #print('sel has indexes, rows, cols, data:')
        #for index in sel.indexes():
        #    print(index, index.row(), index.column(), index.data())
        sm.select(sel, flag)
        #print('end select loop')
        '''
        # constantly scrolling to selection slows everything quite noticeably, especially
        # when using the spike selection sortwin.slider
        if scrollTo and on and len(rows) > 0: # scroll to last row that was just selected
            self.scrollTo(m.index(rows[-1]))
        '''
    def selectedRows(self):
        """Return list of selected rows"""
        return [ i.row() for i in self.selectedIndexes() ]

    def rowSelected(self, row):
        """Simple way to check if a row is selected"""
        return self.model().index(row) in self.selectedIndexes()

    def get_nrowsSelected(self):
        return len(self.selectedIndexes())

    nrowsSelected = property(get_nrowsSelected)

    def selectRandom(self, start, stop, nsamples):
        """Select random sample of rows"""
        start = max(0, start)
        if stop == -1:
            stop = self.nrows
        stop = min(self.nrows, stop)
        nrows = stop - start
        nsamples = min(nsamples, nrows)
        rows = random.sample(xrange(start, stop), nsamples)
        self.selectRows(rows, scrollTo=False)


class NList(SpykeListView):
    """Neuron list view"""
    def __init__(self, parent):
        SpykeListView.__init__(self, parent)
        self.setModel(NListModel(parent))
        self.setItemDelegate(NListDelegate(parent))
        #self.connect(self, QtCore.SIGNAL("activated(QModelIndex)"),
        #             self.on_actionItem_triggered)
        # alternate style of connecting signals, seems "activated" is needed now with
        # new style signals and slots, instead of "triggered", even though "activated"
        # is supposed to be deprecated:
        self.activated.connect(self.on_actionItem_triggered)

    def selectionChanged(self, selected, deselected):
        SpykeListView.selectionChanged(self, selected, deselected, prefix='n')
        selnids = [ i.data().toInt()[0] for i in self.selectedIndexes() ]
        #if 1 <= len(selnids) <= 3: # populate nslist if exactly 1, 2 or 3 neurons selected
        self.sortwin.nslist.neurons = [ self.sortwin.sort.neurons[nid] for nid in selnids ]
        #else:
        #    self.sortwin.nslist.neurons = []

    def on_actionItem_triggered(self, index):
        sw = self.sortwin
        sw.parent().ui.plotButton.click()


class NSList(SpykeListView):
    """Spike list view"""
    def __init__(self, parent):
        SpykeListView.__init__(self, parent)
        self.setModel(NSListModel(parent))
        #self.connect(self, QtCore.SIGNAL("activated(QModelIndex)"),
        #            self.on_actionItem_triggered)
        self.activated.connect(self.on_actionItem_triggered)

    def selectionChanged(self, selected, deselected):
        SpykeListView.selectionChanged(self, selected, deselected, prefix='s')

    def on_actionItem_triggered(self, index):
        sw = self.sortwin
        if sw.sort.stream.is_open():
            sid = self.sids[index.row()]
            spike = sw.sort.spikes[sid]
            sw.parent().seek(spike['t'])
        else:
            sw.parent().ui.plotButton.click()

    def get_neurons(self):
        return self.model().neurons

    def set_neurons(self, neurons):
        """Every time neurons are set, clear any existing selection and update data model"""
        self.clearSelection() # remove any plotted sids, at least for now
        self.model().neurons = neurons

    neurons = property(get_neurons, set_neurons)

    def get_nids(self):
        return np.asarray([ neuron.id for neuron in self.model().neurons ])

    nids = property(get_nids)

    def get_sids(self):
        return self.model().sids

    sids = property(get_sids)

    def keyPressEvent(self, event):
        sw = self.sortwin
        key = event.key()
        # passing horizontal keys to nlist assumes nslist is a single column
        # and are therefore not needed:
        if key in [Qt.Key_Enter, Qt.Key_Return]:
            sw.nlist.keyPressEvent(event) # pass on to nlist
        else:
            SpykeListView.keyPressEvent(self, event) # handle it as usual

    def selectRandom(self, nsamples):
        """Select up to nsamples random rows per neuron"""
        if self.model().sliding == True:
            self.neurons = self.neurons # trigger NSListModel.set_neurons() call
            self.model().sliding = False
        for neuron in self.neurons:
            allrows = self.sids.searchsorted(neuron.sids)
            nsamples = min(nsamples, len(allrows))
            rows = random.sample(allrows, nsamples)
            self.selectRows(rows, scrollTo=False)


class USList(SpykeListView):
    """Unsorted spike list view"""
    def __init__(self, parent):
        SpykeListView.__init__(self, parent)
        self.setModel(USListModel(parent))
        #self.connect(self, QtCore.SIGNAL("activated(QModelIndex)"),
        #             self.on_actionItem_triggered)
        self.activated.connect(self.on_actionItem_triggered)

    def keyPressEvent(self, event):
        sw = self.sortwin
        key = event.key()
        if key in [Qt.Key_Enter, Qt.Key_Return]:
            sw.nlist.keyPressEvent(event) # pass on to nlist
        else:
            SpykeListView.keyPressEvent(self, event) # handle it as usual

    def selectionChanged(self, selected, deselected):
        SpykeListView.selectionChanged(self, selected, deselected, prefix='s')

    def on_actionItem_triggered(self, index):
        sw = self.sortwin
        if sw.sort.stream.is_open():
            sid = sw.sort.usids[index.row()]
            spike = sw.sort.spikes[sid]
            sw.parent().seek(spike['t'])
        else:
            sw.parent().ui.plotButton.click()

    def selectRandom(self, nsamples):
        """Select up to nsamples random rows"""
        SpykeListView.selectRandom(self, 0, -1, nsamples)


class SpykeAbstractListModel(QtCore.QAbstractListModel):
    def __init__(self, parent):
        QtCore.QAbstractListModel.__init__(self, parent)
        self.sortwin = parent

    def updateAll(self):
        """Emit dataChanged signal so that view updates itself immediately.
        Hard to believe this doesn't already exist in some form"""
        i0 = self.createIndex(0, 0) # row, col
        i1 = self.createIndex(self.rowCount()-1, 0) # seems this isn't necessary
        # seems to refresh all, though should only refresh 1st row:
        #self.dataChanged.emit(i0, i0)
        self.dataChanged.emit(i0, i1) # refresh all


class NListModel(SpykeAbstractListModel):
    """Model for neuron list view"""
    def rowCount(self, parent=None):
        try:
            # update nlist tooltip before returning, only +ve nids count as neurons:
            sort = self.sortwin.sort
            neurons = sort.neurons
            nneurons = (np.asarray(sort.norder) > 0).sum()
            goodnids = sort.get_good()
            ngood = len(goodnids)
            ngoodspikes = sum(neurons[nid].nspikes for nid in goodnids)
            self.sortwin.nlist.setToolTip("Neuron list\n"
                                          "%d neurons\n"
                                          "%d good with %d spikes"
                                          % (nneurons, ngood, ngoodspikes))
            return len(sort.norder)
        except AttributeError: # sort doesn't exist
            self.sortwin.nlist.setToolTip("Neuron list")
            return 0

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            neurons = self.sortwin.sort.neurons
            norder = self.sortwin.sort.norder
            try:
                nid = norder[index.row()]
            except IndexError:
                print('WARNING: tried to index non-existent row %d' % index.row())
            #print('.data(): row=%d, val=%d' % (index.row(), nid))
            if role == Qt.DisplayRole:
                return nid # no need to use QVariant() apparently
            elif role == Qt.ToolTipRole:
                neuron = neurons[nid]
                try:
                    chan = neuron.chan
                except ValueError: # probably not enough overlapping chans for a template
                    chan = None
                pos = neuron.cluster.pos
                return ('nid: %d\n' % nid +
                        '%d spikes\n' % neuron.nspikes +
                        'chan: %r\n' % chan +
                        't: %d us\n' % pos['t'] +
                        'dt: %.4g us\n' % pos['dt'] +
                        'x0: %.4g um\n' % pos['x0'] +
                        'y0: %.4g um\n' % pos['y0'] +
                        'Vpp: %.4g uV\n' % pos['Vpp'] +
                        'sx: %.4g um' % pos['sx'])
            # this stuff is handled in NListDelegate:
            '''
            elif role == Qt.ForegroundRole:
                if nid in self.sortwin.sort.get_good():
                    return QtGui.QBrush(QtGui.QColor(255, 255, 255))
            elif role == Qt.BackgroundRole:
                if nid in self.sortwin.sort.get_good():
                    return QtGui.QBrush(QtGui.QColor(0, 128, 0))
            '''
class SListModel(SpykeAbstractListModel):
    """Base model for spike list models"""
    def spiketooltip(self, spike):
        return ('sid: %d\n' % spike['id'] +
                'nid: %d\n' % spike['nid'] +
                'chan: %d\n' % spike['chan'] +
                't: %d us\n' % spike['t'] +
                'dt: %.4g us\n' % spike['dt'] +
                'x0: %.4g um\n' % spike['x0'] +
                'y0: %.4g um\n' % spike['y0'] +
                'Vpp: %.4g uV\n' % spike['Vpp'] +
                'sx: %.4g um' % spike['sx'])


class NSListModel(SListModel):
    """Model for neuron spikes list view"""
    def __init__(self, parent):
        SpykeAbstractListModel.__init__(self, parent)
        self._neurons = []
        self.nspikes = 0
        self.sids = np.empty(0, dtype=np.int32)

    def get_neurons(self):
        return self._neurons

    def set_neurons(self, neurons):
        self._neurons = neurons
        if neurons:
            self.sids = np.concatenate([ neuron.sids for neuron in neurons ])
            self.sids.sort() # keep them sorted
            self.sortwin.slider.setEnabled(True)
        else:
            self.sids = np.empty(0, dtype=np.int32)
            self.sortwin.slider.setEnabled(False)
        self.nspikes = len(self.sids)
        # triggers new calls to rowCount() and data(), and critically, clears selection
        # before moving slider to pos 0, which triggers slider.valueChanged:
        self.reset()
        self.sortwin.slider.setValue(0) # reset position to 0
        self.sortwin.update_slider() # update limits and step sizes
        self.sliding = False

    neurons = property(get_neurons, set_neurons)

    def rowCount(self, parent=None):
        # update nslist tooltip before returning:
        self.sortwin.nslist.setToolTip("Sorted spike list\n%d spikes" % self.nspikes)
        return self.nspikes

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in [Qt.DisplayRole, Qt.ToolTipRole]:
            sid = int(self.sids[index.row()])
            if role == Qt.DisplayRole:
                return sid
            elif role == Qt.ToolTipRole:
                spike = self.sortwin.sort.spikes[sid]
                return self.spiketooltip(spike)


class USListModel(SListModel):
    """Model for unsorted spike list view"""
    def rowCount(self, parent=None):
        try:
            nspikes = len(self.sortwin.sort.usids)
            # update uslist tooltip before returning:
            self.sortwin.uslist.setToolTip("Unsorted spike list\n%d spikes" % nspikes)
            return nspikes
        except AttributeError: # sort doesn't exist
            self.sortwin.uslist.setToolTip("Unsorted spike list")
            return 0

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in [Qt.DisplayRole, Qt.ToolTipRole]:
            sid = int(self.sortwin.sort.usids[index.row()])
            if role == Qt.DisplayRole:
                return sid
            elif role == Qt.ToolTipRole:
                spike = self.sortwin.sort.spikes[sid]
                return self.spiketooltip(spike)


class NListDelegate(QtGui.QStyledItemDelegate):
    """Delegate for neuron list view, modifies appearance of items"""
    def __init__(self, parent):
        QtGui.QStyledItemDelegate.__init__(self, parent)
        self.sortwin = parent
        palette = QtGui.QApplication.palette()
        self.selectedgoodbrush = QtGui.QBrush(QtGui.QColor(0, 0, 255)) # blue
        self.unselectedgoodbrush = QtGui.QBrush(QtGui.QColor(0, 128, 0)) # mid green
        self.selectedbrush = palette.highlight()
        self.unselectedbrush = palette.base()
        self.selectedgoodpen = QtGui.QPen(Qt.white)
        self.unselectedgoodpen = QtGui.QPen(Qt.white)
        self.selectedpen = QtGui.QPen(palette.highlightedText().color())
        self.unselectedpen = QtGui.QPen(palette.text().color())
        self.focusedpen = QtGui.QPen(Qt.gray, 0, Qt.DashLine)
        self.focusedpen.setDashPattern([1, 1])
        self.focusedpen.setCapStyle(Qt.FlatCap)

    def paint(self, painter, option, index):
        """Change background colour for nids designated as "good"""
        model = index.model()
        nid = model.data(index) # should come out as an int
        good = nid in self.sortwin.sort.get_good()
        # don't care whether self is active or inactive, only care about
        # selection, "good", and focused states
        selected = option.state & QtGui.QStyle.State_Selected
        focused = option.state & QtGui.QStyle.State_HasFocus
        painter.save()
        # paint background:
        painter.setPen(QtGui.QPen(Qt.NoPen))
        if selected:
            if good:
                painter.setBrush(self.selectedgoodbrush)
            else: # use default selection brush
                painter.setBrush(self.selectedbrush)
        else: # unselected
            if good:
                painter.setBrush(self.unselectedgoodbrush)
            else: # use default background brush
                painter.setBrush(self.unselectedbrush)
        painter.drawRect(option.rect)
        # paint focus rect:
        if focused:
            rect = copy(option.rect)
            painter.setBrush(Qt.NoBrush) # no need to draw bg again
            painter.setPen(self.focusedpen)
            rect.adjust(0, 0, -1, -1) # make space for outline
            painter.drawRect(rect)
        # paint foreground:
        value = index.data(Qt.DisplayRole)
        if selected:
            if good:
                painter.setPen(self.selectedgoodpen)
            else: # use default selection pen
                painter.setPen(self.selectedpen)
        else: # unselected
            if good:
                painter.setPen(self.unselectedgoodpen)
            else: # use default background pen
                painter.setPen(self.unselectedpen)
        text = value.toString()
        painter.drawText(option.rect, Qt.AlignCenter, text)
        painter.restore()


class ClusterTabSpinBox(QtGui.QSpinBox):
    """Intercept CTRL+Z key event for cluster undo instead of spinbox edit undo"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.topLevelWidget().on_actionUndo_triggered()
        else:
            QtGui.QSpinBox.keyPressEvent(self, event) # handle it as usual


class ClusterTabDoubleSpinBox(QtGui.QDoubleSpinBox):
    """Intercept CTRL+Z key event for cluster undo instead of spinbox edit undo"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.topLevelWidget().on_actionUndo_triggered()
        else:
            QtGui.QDoubleSpinBox.keyPressEvent(self, event) # handle it as usual


class ClusteringGroupBox(QtGui.QGroupBox):
    """Make ENTER key event activate the cluster button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().ui.clusterButton.click()
        else:
            QtGui.QGroupBox.keyPressEvent(self, event) # handle it as usual


class PlottingGroupBox(QtGui.QGroupBox):
    """Make ENTER key event activate the plot button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().ui.plotButton.click()
        else:
            QtGui.QGroupBox.keyPressEvent(self, event) # handle it as usual


class XCorrsGroupBox(QtGui.QGroupBox):
    """Make ENTER key event activate the correlograms plot button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().ui.plotXcorrsButton.click()
        else:
            QtGui.QGroupBox.keyPressEvent(self, event) # handle it as usual


class SpikeSelectionSlider(QtGui.QSlider):
    """Make ENTER key event activate the plot button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().spykewindow.ui.plotButton.click()
        else:
            QtGui.QSlider.keyPressEvent(self, event) # handle it as usual


class Stack(list):
    """A list that doesn't allow -ve indices"""
    def __getitem__(self, key):
        if key < 0:
            raise IndexError('stack index %d out of range' % key)
        return list.__getitem__(self, key)


class ClusterChange(object):
    """Stores info for undoing/redoing a change to any set of clusters"""
    def __init__(self, sids, spikes, message):
        self.sids = sids
        self.spikes = spikes
        self.message = message

    def __repr__(self):
        return self.message

    def save_old(self, oldclusters, oldnorder, oldgood):
        self.oldnids = self.spikes['nid'][self.sids] # this seems to create a copy
        self.oldunids = [ c.id for c in oldclusters ]
        self.oldposs = [ c.pos.copy() for c in oldclusters ]
        self.oldnormposs = [ c.normpos.copy() for c in oldclusters ]
        self.oldnorder = copy(oldnorder)
        self.oldgood = copy(oldgood)

    def save_new(self, newclusters, newnorder, newgood):
        self.newnids = self.spikes['nid'][self.sids] # this seems to create a copy
        self.newunids = [ c.id for c in newclusters ]
        self.newposs = [ c.pos.copy() for c in newclusters ]
        self.newnormposs = [ c.normpos.copy() for c in newclusters ]
        self.newnorder = copy(newnorder)
        self.newgood = copy(newgood)

def get_sha1(fname, blocksize=2**20):
    """Gets the sha1 hash of file designated by fname (with full path)"""
    m = hashlib.sha1()
    with open(fname, 'rb') as f:
        # continually update hash until EOF
        while True:
            block = f.read(blocksize)
            if not block:
                break
            m.update(block)
    return m.hexdigest()

def intround(n):
    """Round to the nearest integer, return an integer. Works on arrays.
    Saves on parentheses, nothing more"""
    if iterable(n): # it's a sequence, return as an int64 array
        return np.int64(np.round(n))
    else: # it's a scalar, return as normal Python int
        return int(round(n))

def intfloor(n):
    """Round down to the nearest integer, return an integer. Works on arrays.
    Saves on parentheses, nothing more"""
    if iterable(n): # it's a sequence, return as an int64 array
        return np.int64(np.floor(n))
    else: # it's a scalar, return as normal Python int
        return int(np.floor(n))

def intceil(n):
    """Round up to the nearest integer, return an integer. Works on arrays.
    Saves on parentheses, nothing more"""
    if iterable(n): # it's a sequence, return as an int64 array
        return np.int64(np.ceil(n))
    else: # it's a scalar, return as normal Python int
        return int(np.ceil(n))

def iterable(x):
    """Check if the input is iterable, stolen from numpy.iterable()"""
    try:
        iter(x)
        return True
    except TypeError:
        return False

def toiter(x):
    """Convert to iterable. If input is iterable, returns it. Otherwise returns it in a list.
    Useful when you want to iterate over something (like in a for loop),
    and you don't want to have to do type checking or handle exceptions
    when it isn't a sequence"""
    if iterable(x):
        return x
    else:
        return [x]

def tocontig(x):
    """Return C contiguous copy of array x if it isn't C contiguous already"""
    if not x.flags.c_contiguous:
        x = x.copy()
    return x
'''
# use np.vstack instead:
def cvec(x):
    """Return x as a column vector. x must be a scalar or a vector"""
    x = np.asarray(x)
    assert x.squeeze().ndim in [0, 1]
    try:
        nrows = len(x)
    except TypeError: # x is scalar?
        nrows = 1
    x.shape = (nrows, 1)
    return x
'''
def is_empty(x):
    """Check if sequence is empty. There really should be a np.is_empty function"""
    print("WARNING: not thoroughly tested!!!")
    x = np.asarray(x)
    if np.prod(x.shape) == 0:
        return True
    else:
        return False

def cut(ts, trange):
    """Returns timestamps, where tstart <= timestamps <= tend
    Copied and modified from neuropy rev 149"""
    lo, hi = argcut(ts, trange)
    return ts[lo:hi] # slice it

def argcut(ts, trange):
    """Returns timestamp slice indices, where tstart <= timestamps <= tend
    Copied and modified from neuropy rev 149"""
    tstart, tend = trange[0], trange[1]
    '''
    # this is what we're trying to do:
    return ts[ (ts >= tstart) & (ts <= tend) ]
    ts.searchsorted([tstart, tend]) method does it faster, because it assumes ts are ordered.
    It returns an index where the values would fit in ts. The index is such that
    ts[index-1] < value <= ts[index]. In this formula ts[ts.size]=inf and ts[-1]= -inf
    '''
    lo, hi = ts.searchsorted([tstart, tend]) # indices where tstart and tend would fit in ts
    # can probably avoid all this end inclusion code by using the 'side' kwarg,
    # not sure if I want end inclusion anyway:
    '''
    if tend == ts[min(hi, len(ts)-1)]:
        # if tend matches a timestamp (protect from going out of index bounds when checking)
        hi += 1 # inc to include a timestamp if it happens to exactly equal tend.
                # This gives us end inclusion
        hi = min(hi, len(ts)) # limit hi to max slice index (==max value index + 1)
    '''
    return lo, hi

def dist(a, b):
    """Return the Euclidean distance between two N-dimensional coordinates"""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(((a-b)**2).sum())

def eucd(coords):
    """Generates Euclidean distance matrix from a
    sequence of n m-dimensional coordinates. Nice and fast.
    Written by Willi Richert
    Taken from:
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/498246
    on 2006/11/11
    """
    coords = np.asarray(coords)
    n, m = coords.shape
    delta = np.zeros((n, n), dtype=np.float64)
    for d in xrange(m):
        data = coords[:, d]
        delta += (data - data[:, np.newaxis]) ** 2
    return np.sqrt(delta)

def revcmp(x, y):
    """Does the reverse of cmp():
    Return negative if y<x, zero if y==x, positive if y>x"""
    return cmp(y, x)


class Gaussian(object):
    """Gaussian function, works with ndarray inputs"""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        """Called when self is called as a f'n.
        Don't bother normalizing by 1/(sigma*np.sqrt(2*pi)),
        don't care about normalizing the integral,
        just want to make sure that f(0) == 1"""
        return np.exp( -(x-self.mu)**2 / (2*self.sigma**2) )

    def __getitem__(self, x):
        """Called when self is indexed into"""
        return self(x)


def g(x0, sx, x):
    """1-D Gaussian"""
    return np.exp( -(x-x0)**2 / (2*sx**2) )

def g2(x0, y0, sx, sy, x, y):
    """2-D Gaussian"""
    arg = -(x-x0)**2 / (2*sx**2) - (y-y0)**2 / (2*sy**2)
    return np.exp(arg)

def g3(x0, y0, z0, sx, sy, sz, x, y, z):
    """3-D Gaussian"""
    return np.exp( -(x-x0)**2 / (2*sx**2) - (y-y0)**2 / (2*sy**2) - (z-z0)**2 / (2*sz**2) )

def cauchy(x0, gx, x):
    """1-D Cauchy. See http://en.wikipedia.org/wiki/Cauchy_distribution"""
    #return INVPI * gx/((x-x0)**2+gx**2)
    gx2 = gx * gx
    return gx2 / ((x-x0)**2 + gx2)

def cauchy2(x0, y0, gx, gy, x, y):
    """2-D Cauchy"""
    #return INVPI * gx/((x-x0)**2+gx**2) * gy/((y-y0)**2+gy**2)
    return (gx*gy)**2 / ((x-x0)**2 + gx**2) / ((y-y0)**2 + gy**2)

def Vf(Im, x0, y0, z0, sx, sy, sz, x, y, z):
    """1/r voltage decay function in 2D space
    What to do with the singularity so that the leastsq gets a smooth differentiable f'n?"""
    #if np.any(x == x0) and np.any(y == y0) and np.any(z == z0):
    #    raise ValueError, 'V undefined at singularity'
    return Im / (4*pi) / np.sqrt( sx**2 * (x-x0)**2 + sy**2 * (y-y0)**2 + sz**2 * (z-z0)**2)

def dgdmu(mu, sigma, x):
    """Partial of g wrt mu"""
    return (x - mu) / sigma**2 * g(mu, sigma, x)

def dgdsigma(mu, sigma, x):
    """Partial of g wrt sigma"""
    return (x**2 - 2*x*mu + mu**2) / sigma**3 * g(mu, sigma, x)

def dg2dx0(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt x0"""
    return g(y0, sy, y) * dgdmu(x0, sx, x)

def dg2dy0(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt y0"""
    return g(x0, sx, x) * dgdmu(y0, sy, y)

def dg2dsx(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt sx"""
    return g(y0, sy, y) * dgdsigma(x0, sx, x)

def dg2dsy(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt sy"""
    return g(x0, sx, x) * dgdsigma(y0, sy, y)

def RM(theta):
    """Return 2D (2x2) rotation matrix, with theta counterclockwise rotation in radians"""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class Poo(object):
    """Poo function, works with ndarray inputs"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        """Called when self is called as a f'n"""
        return (1+self.a*x) / (self.b+self.c*x**2)

    def __getitem__(self, x):
        """Called when self is indexed into"""
        return self(x)


def hamming(t, N):
    """Return y values of Hamming window at sample points t"""
    #if N == None:
    #    N = (len(t) - 1) / 2
    return 0.54 - 0.46 * np.cos(pi * (2*t + N)/N)

def hex2rgb(hexcolours):
    """Convert colours RGB hex string list into an RGB int array"""
    hexcolours = toiter(hexcolours)
    rgb = []
    for s in hexcolours:
        s = s[len(s)-6:len(s)] # get last 6 characters
        r, g, b = s[0:2], s[2:4], s[4:6]
        r, g, b = int(r, base=16), int(g, base=16), int(b, base=16)
        rgb.append((r, g, b))
    return np.uint8(rgb)

def hex2rgba(hexcolours, alpha=255):
    """Convert colours RGB hex string list into an RGBA int array"""
    assert type(alpha) == int and 0 <= alpha <= 255
    rgb = hex2rgb(hexcolours)
    alphas = np.repeat(alpha, len(rgb))
    alphas.shape = -1, 1 # make it 2D column vector
    return np.concatenate([rgb, alphas], axis=1)

def hex2floatrgba(hexcolours, alpha=255):
    """Convert colours RGB hex string list into an RGBA float array"""
    assert type(alpha) == int and 0 <= alpha <= 255
    rgba = hex2rgba(hexcolours, alpha)
    return np.float64(rgba) / 255.

def rgb2hex(rgbcolours):
    """Convert RGB int array into a hex string list"""
    rgbcolours = toiter(rgbcolours)
    hx = []
    for rgb in rgbcolours:
        r, g, b = rgb
        h = hex(r*2**16 + g*2**8 + b)
        h = lrstrip(h, '0x', 'L')
        pad = (6 - len(h)) * '0'
        h = '#' + pad + h
        hx.append(h)
    return hx

c = np.cos
s = np.sin

def Rx(t):
    """Rotation matrix around x axis, theta in radians"""
    return np.matrix([[1, 0,     0   ],
                      [0, c(t), -s(t)],
                      [0, s(t),  c(t)]])

def Ry(t):
    """Rotation matrix around y axis, theta in radians"""
    return np.matrix([[ c(t), 0, s(t)],
                      [ 0,    1, 0   ],
                      [-s(t), 0, c(t)]])

def Rz(t):
    """Rotation matrix around z axis, theta in radians"""
    return np.matrix([[c(t), -s(t), 0],
                      [s(t),  c(t), 0],
                      [0,     0,    1]])

def R(tx, ty, tz):
    """Return full 3D rotation matrix, given thetas in degress.
    Mayavi (tvtk actually) rotates axes in Z, X, Y order, for
    some unknown reason. So, we have to do the same. See:
    tvtk_classes.zip/actor.py:32
    tvtk_classes.zip/prop3d.py:67
    """
    # convert to radians, then take matrix product
    return Rz(tz*pi/180)*Rx(tx*pi/180)*Ry(ty*pi/180)
'''
def normdeg(angle):
    return angle % 360

def win2posixpath(path):
    path = path.replace('\\', '/')
    path = os.path.splitdrive(path)[-1] # remove drive name from start
    return path

def oneD2D(a):
    """Convert 1D array to 2D array. Can do this just as easily using a[None, :]"""
    a = a.squeeze()
    assert a.ndim == 1, "array has more than one non-singleton dimension"
    a.shape = 1, len(a) # make it 2D
    return a

def twoD1D(a):
    """Convert trivially 2D array to 1D array. Seems unnecessary. Just call squeeze()"""
    a = a.squeeze()
    assert a.ndim == 1, "array has more than one non-singleton dimension"
    return a
'''
def is_unique(a):
    """Check whether a has purely unique values in it"""
    u = np.unique(a)
    if len(a) != len(u):
        return False
    else:
        return True

def intersect1d(arrays, assume_unique=False):
    """Find the intersection of any number of 1D arrays.
    Return the sorted, unique values that are in all of the input arrays.
    Adapted from numpy.lib.arraysetops.intersect1d"""
    N = len(arrays)
    if N == 0:
        return np.asarray(arrays)
    arrays = list(arrays) # allow assignment
    if not assume_unique:
        for i, arr in enumerate(arrays):
            arrays[i] = np.unique(arr)
    aux = np.concatenate(arrays) # one long 1D array
    aux.sort() # sorted
    if N == 1:
        return aux
    shift = N-1
    return aux[aux[shift:] == aux[:-shift]]

def rowtake(a, i):
    """For each row in a, return values according to column indices in the
    corresponding row in i. Returned shape == i.shape"""
    assert a.ndim == 2
    assert i.ndim <= 2
    '''
    if i.ndim == 1:
        j = np.arange(a.shape[0])
    else: # i.ndim == 2
        j = np.repeat(np.arange(a.shape[0]), i.shape[1])
        j.shape = i.shape
    j *= a.shape[1]
    j += i
    return a.flat[j]
    '''
    # this is about 3X faster:
    if i.ndim == 1:
        return a[np.arange(a.shape[0]), i]
    else: # i.ndim == 2
        return a[np.arange(a.shape[0])[:, None], i]

def td2usec(td):
    """Convert datetime.timedelta to int microseconds"""
    sec = td.total_seconds() # float
    usec = intround(sec * 1000000) # round to nearest us
    return usec

def td2fusec(td):
    """Convert datetime.timedelta to float microseconds"""
    sec = td.total_seconds() # float
    return sec * 1000000

def td2days(td):
    """Convert datetime.timedelta to days"""
    sec = td.total_seconds() # float
    days = sec / 3600 / 24
    return days

def unsortedis(x):
    """Return indices of entries in x that are out of order"""
    x = np.asarray(x)
    try:
        if x.dtype.kind == 'u':
            # x is unsigned int array, risk of int underflow in np.diff
            x = np.int64(x)
    except AttributeError:
        pass # no dtype, not an array
    return np.where(np.diff(x) < 0)[0] # where is the diff between consecutive entries < 0?

def issorted(x):
    """Check if x is sorted"""
    return len(unsortedis(x)) == 0
    # or, you could compare the array to an explicitly sorted version of itself,
    # and see if they're identical

def concatenate_destroy(arrs):
    """Concatenate list of arrays along 0th axis, destroying them in the process.
    Doesn't duplicate everything in arrays, as does numpy.concatenate. Only
    temporarily duplicates one array at a time, saving memory"""
    if type(arrs) != list:
        raise TypeError('arrays must be in a list')
    #arrs = list(arrs) # don't do this! this prevents destruction of the original arrs
    nrows = 0
    subshape = arrs[0].shape[1::] # dims excluding concatenation dim
    dtype = arrs[0].dtype
    # ensure all arrays in arrs are compatible:
    for i, a in enumerate(arrs):
        nrows += len(a)
        if a.shape[1::] != subshape:
            raise TypeError("array %d has subshape %r instead of %r" %
                           (i, a.shape[1::], subshape))
        if a.dtype != dtype:
            raise TypeError("array %d has dtype %r instead of %r" % (i, a.dtype, dtype))
    subshape = list(subshape)
    shape = [nrows] + subshape

    # unlike np.zeros, it seems np.empty doesn't allocate real memory, but does temporarily
    # allocate virtual memory, which is then converted to real memory as 'a' is filled:
    try:
        a = np.empty(shape, dtype=dtype) # empty only allocates virtual memory
    except MemoryError:
        raise MemoryError("concatenate_destroy: not enough virtual memory to allocate "
                          "destination array. Create/grow your swap file?")
        
    rowi = 0
    for i in range(len(arrs)):
        arr = arrs.pop(0)
        nrows = len(arr)
        a[rowi:rowi+nrows] = arr # concatenate along 0th axis
        rowi += nrows
    return a

def lst2shrtstr(lst, sigfigs=4, brackets=False):
    """Return string representation of list, replacing any floats with potentially
    shorter representations with fewer sig figs. Any string items in list will be
    simplified by having their quotes removed"""
    gnumfrmt = string.join(['%.', str(sigfigs), 'g'], sep='')
    strlst = []
    for val in lst:
        try:
            strlst.append(gnumfrmt % val)
        except TypeError:
            strlst.append(val) # val isn't a general number
    s = string.join(strlst, sep=', ')
    if brackets:
        s = string.join(['[', s, ']'], sep='')
    return s

def rms(a, axis=None):
    """Return root-mean-squared value of array a along axis"""
    return np.sqrt(np.mean(a**2, axis))

def rmserror(a, b, axis=None):
    """Return root-mean-squared error between arrays a and b"""
    return rms(a - b, axis=axis)

def lstrip(s, strip):
    """What I think str.lstrip should really do"""
    if s.startswith(strip):
        return s[len(strip):] # strip it
    else:
        return s

def rstrip(s, strip):
    """What I think str.rstrip should really do"""
    if s.endswith(strip):
        return s[:-len(strip)] # strip it
    else:
        return s

def strip(s, strip):
    """What I think str.strip should really do"""
    return rstrip(lstrip(s, strip), strip)

def lrstrip(s, lstr, rstr):
    """Strip lstr from start of s and rstr from end of s"""
    return rstrip(lstrip(s, lstr), rstr)

def isascii(c):
    """Check if character c is a printable character, TAB, LF, or CR"""
    d = ord(c) # decimal representation
    return 32 <= d <= 127 or d in [9, 10, 13]

def rstripnonascii(s):
    """Return a new string with all characters after the first non-ASCII character
    stripped from the string"""
    for i, c in enumerate(s):
        if not isascii(c):
            return s[:i]
    return s

def pad(x, align=8):
    """Pad x with null bytes so it's a multiple of align bytes long"""
    if type(x) == str: # or maybe unicode?
        return padstr(x, align=align)
    elif type(x) == np.ndarray:
        return padarr(x, align=align)
    else:
        raise TypeError('Unhandled type %r in pad()')

def padstr(x, align=8):
    """Pad string x with null bytes so it's a multiple of align bytes long"""
    nbytes = len(x)
    rem = nbytes % align
    npadbytes = align - rem if rem else 0 # nbytes to pad with for 8 byte alignment
    if npadbytes == 0:
        return x
    x = x.encode('ascii') # ensure it's pure ASCII, where each char is 1 byte
    x += '\0' * npadbytes # returns a copy, doesn't modify in place
    assert len(x) % align == 0
    return x

def padarr(x, align=8):
    """Flatten array x and pad with null bytes so it's a multiple of align bytes long"""
    nitems = len(x.ravel())
    nbytes = x.nbytes
    dtypenbytes = x.dtype.itemsize
    rem = nbytes % align
    npadbytes = align - rem if rem else 0 # nbytes to pad with for 8 byte alignment
    if npadbytes == 0:
        return x
    if npadbytes % dtypenbytes != 0:
        raise RuntimeError("Can't pad %d byte array to %d byte alignment" %
                           (dtypenbytes, align))
    npaditems = npadbytes / dtypenbytes
    x = x.ravel().copy() # don't modify in place
    # pads with npaditems zeros, each of length dtypenbytes:
    x.resize(nitems + npaditems, refcheck=False)
    assert x.nbytes % align == 0
    return x

def shiftpad(a, n):
    """Horizontally shift 2D array a *in-place* by n points. -ve n shifts
    left, +ve shifts right. Pad with edge values at the appropriate end.
    This is probably the same as np.roll(), except edge values are padded
    instead of wrapped. Also, I think np.roll returns a copy"""
    assert a.ndim == 2
    assert type(n) == int
    assert n != 0
    if n > 0: # shift right, pad with left edge
        ledge = a[:, 0, None] # keep it 2D (nrows x 1)
        a[:, n:] = a[:, :-n] # throw away right edge
        a[:, 1:n] = ledge # pad with left edge
    else: # n < 0, shift left, pad with right edge
        redge = a[:, -1, None] # keep it 2D (nrows x 1)
        a[:, :n] = a[:, -n:] # throw away left edge
        a[:, n:-1] = redge # pad with right edge
    # no need to return anything

def rollwin(a, width):
    """Return a.nd + 1 dimensional array, where the last dimension contains
    consecutively shifted windows of a of the given width, each shifted by 1
    along the last dimension of a. This allows for calculating rolling stats,
    as well as searching for the existence and position of subarrays in a
    larger array, all without having to resort to Python loops or making
    copies of a.

    Taken from:
        http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
        http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
        http://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy

    Ex 1:
    >>> x = np.arange(10).reshape((2,5))
    >>> rollwin(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])    
    >>> np.mean(rollwin(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    Ex 2:
    >>> a = np.arange(10)
    >>> np.random.shuffle(a)
    >>> a
    array([7, 3, 6, 8, 4, 0, 9, 2, 1, 5])
    >>> rollwin(a, 3) == [8, 4, 0]
    array([[False, False, False],
           [False, False, False],
           [False, False, False],
           [ True,  True,  True],
           [False, False, False],
           [False, False, False],
           [False, False, False],
           [False, False, False]], dtype=bool)
    >>> np.all(rollwin(a, 3) == [8, 4, 0], axis=1)
    array([False, False, False,  True, False, False, False, False], dtype=bool)
    >>> np.where(np.all(rollwin(a, 3) == [8, 4, 0], axis=1))[0][0]
    3
    """
    shape = a.shape[:-1] + (a.shape[-1] - width + 1, width)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rollwin2D(a, width):
    """A modified version of rollwin. Allows for easy columnar search of 2D
    subarray b within larger 2D array a, assuming both have the same number of
    rows.
    
    Ex:
    >>> a
    array([[44, 89, 34, 67, 11, 92, 22, 72, 10, 81],
           [52, 40, 29, 35, 67, 10, 24, 23, 65, 51],
           [70, 58, 14, 34, 11, 66, 47, 68, 11, 56],
           [70, 55, 47, 30, 39, 79, 71, 70, 67, 33]])    
    >>> b
    array([[67, 11, 92],
           [35, 67, 10],
           [34, 11, 66],
           [30, 39, 79]])
    >>> np.where((rollwin2D(a, 3) == b).all(axis=1).all(axis=1))[0]
    array([3])
    """
    assert a.ndim == 2
    shape = (a.shape[1] - width + 1, a.shape[0], width)
    strides = (a.strides[-1],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def argcolsubarr2D(a, b):
    """Return column index of smaller subarray b within bigger array a. Both
    must be 2D and have the same number of rows. Raises IndexError if b is not
    a subarray of a"""
    assert a.ndim == b.ndim == 2
    assert a.shape[0] == b.shape[0] # same nrows
    width = b.shape[1] # ncols in b
    return np.where((rollwin2D(a, width) == b).all(axis=1).all(axis=1))[0]

def lrrep2Darrstripis(a):
    """Return left and right slice indices that strip repeated values from all rows
    from left and right ends of 2D array a, such that a[:, lefti:righti] gives you
    the stripped version.

    Ex:
    >>> a
    array([[44, 44, 44, 44, 89, 34, 67, 11, 92, 22, 72, 10, 81, 81, 81],
           [52, 52, 52, 52, 40, 29, 35, 67, 10, 24, 23, 65, 51, 51, 51],
           [70, 70, 70, 70, 58, 14, 34, 11, 66, 47, 68, 11, 56, 56, 56],
           [70, 70, 70, 70, 55, 47, 30, 39, 79, 71, 70, 67, 33, 33, 33]])
    >>> lrrep2Darrstripis(a)
    (3, -2)
    """
    assert a.ndim == 2
    left = a[:, :1] # 2D column vector
    right = a[:, -1:] # 2D column vector
    leftcolis = argcolsubarr2D(a, left)
    lefti = 0 # at least 1 hit, at the far left edge
    if len(leftcolis) > 1: # multiple hits, get slice index of rightmost consecutive hit
        consecis = np.where(np.diff(leftcolis) == 1)[0]
        if len(consecis) > 0:
            lefti = max(consecis) + 1
    rightcolis = argcolsubarr2D(a, right)
    righti = a.shape[1] # at least 1 hit, at the far right edge
    if len(rightcolis) > 1: # multiple hits, get slice index of leftmost consecutive hit
        consecis = np.where(np.diff(rightcolis)[::-1] == 1)[0]
        if len(consecis) > 0:
            righti = -(max(consecis) + 1)
    return lefti, righti

def normpdf(p, lapcorrect=1e-10):
    """Ensure p is normalized (sums to 1). Return p unchanged if it's already normalized.
    Otherwise, return it normalized. I guess this treats p as a pmf, not strictly a pdf.
    Optional apply Laplacian correction to avoid 0s"""
    p = np.float64(p) # copy and ensure it's float before modifying in-place
    if lapcorrect and (p == 0).any():
        p += lapcorrect
    psum = p.sum()
    if not np.allclose(psum, 1.0) and psum > 0: # make sure the probs sum to 1
        #print("p sums to %f instead of 1, normalizing" % psum)
        p /= psum
    return p

def negentropy(x, axis=0):
    """Return estimate of negative entropy (and differential entropy) of ndarray x along axis.
    Adapted from Aapo Hyvarinen's mentappr.m dated May 2012, which is based on his NIPS*97
    paper: http://www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf - "New approximations of
    differential entropy for independent component analysis and projection pursuit"
    """
    # constants:
    k1 = 36 / (8*np.sqrt(3) - 9)
    gamma = 0.37457
    k2 = 79.047
    # entropy of a standard Gaussian, 1.4189 (in bits? maybe not, since it's natural log):
    gaussianEntropy = np.log(2*pi) / 2 + 0.5
    # normalize to 0 mean and unit variance:
    x = x - x.mean(axis=axis) # don't do this in place
    stdx = x.std(axis=axis)
    x = x / stdx

    negentropy = ( k2*((np.log(np.cosh(x))).mean(axis=axis) - gamma)**2 +
                   k1*((x*np.exp(-x**2/2)).mean(axis=axis))**2 )
    #diffentropy = gaussianEntropy - negentropy + np.log(stdx)
    return negentropy

def DKL(p, q):
    """Kullback-Leibler divergence from true probability distribution p to arbitrary
    distribution q"""
    assert len(p) == len(q)
    p, q = normpdf(np.asarray(p)), normpdf(np.asarray(q))
    return sum(p * np.log2(p/q))
    
def DJS(p, q):
    """Jensen-Shannon divergence, a symmetric measure of divergence between
    distributions p and q"""
    assert len(p) == len(q)
    p, q = normpdf(np.asarray(p)), normpdf(np.asarray(q))
    m = (p + q) / 2
    return (DKL(p, m) + DKL(q, m)) / 2

def filter(data, sampfreq=1000, f0=0, f1=7, fr=0.5, gpass=0.01, gstop=30, ftype='ellip'):
    """Bandpass filter data on row indices chanis, between f0 and f1 (Hz), with filter
    rolloff (?) fr (Hz).

    ftype: 'ellip', 'butter', 'cheby1', 'cheby2', 'bessel'
    """
    w0 = f0 / (sampfreq / 2) # fraction of Nyquist frequency == 1/2 sampling rate
    w1 = f1 / (sampfreq / 2)
    wr = fr / (sampfreq / 2)
    if w0 == 0:
        wp = w1
        ws = w1+wr
    elif w1 == 0:
        wp = w0
        ws = w0-wr
    else:
        wp = [w0, w1]
        ws = [w0-wr, w1+wr]
    b, a = scipy.signal.iirdesign(wp, ws, gpass=gpass, gstop=gstop, analog=0, ftype=ftype)
    data = scipy.signal.lfilter(b, a, data)
    return data, b, a

def filterord(data, sampfreq=1000, f0=300, f1=None, order=4, rp=None, rs=None,
              btype='highpass', ftype='butter'):
    """Bandpass filter data by specifying filter order and btype, instead of gpass and gstop.

    btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    ftype: 'ellip', 'butter', 'cheby1', 'cheby2', 'bessel'

    For 'ellip', need to also specify passband and stopband ripple with rp and rs.
    """
    if f0 != None and f1 != None: # both are specified
        assert btype in ['bandpass', 'bandstop']
        fn = np.array([f0, f1])
    elif f0 != None: # only f0 is specified
        assert btype == 'highpass'
        fn = f0
    elif f1 != None: # only f1 is specified
        assert btype == 'lowpass'
        fn = f1
    else: # neither f0 nor f1 are specified
        raise ValueError('at least one of f0 or f1 have to be specified')
    wn = fn / (sampfreq / 2) # wn can be either a scalar or a length 2 vector
    b, a = scipy.signal.iirfilter(order, wn, rp=rp, rs=rs, btype=btype, analog=0,
                                  ftype=ftype, output='ba')
    data = scipy.signal.lfilter(b, a, data)
    return data, b, a

def WMLDR(data, wname="db4", maxlevel=6, mode='sym'):
    """Perform wavelet multi-level decomposition and reconstruction (WMLDR) on multichannel
    data. See Wiltschko2008. Default to Daubechies(4) wavelet. Modifies data in-place, at
    least for now. The effective cutoff frequency is:

    fc = (sampfreq / 2) / 2**maxlevel                     (Wiltschko2008)

    For sampfreq of 25 kHz and maxlevel of 6, the effective cutoff frequency is 195 Hz.
    For sampfreq of 30 kHz and maxlevel of 6, the effective cutoff frequency is 234 Hz.

    TODO: for now, this only returns highpass data. In the future, this probably should
    return both low and highpass data (and not modify it in-place). The Discussion in
    Wiltschko2008 suggests that this approach cannot be used to extract the LFP, but
    I don't see why you can't simply subtract the highpass data from the raw data to get the
    lowpass data.

    Signal extension modes (from PyWavelets docs):

    PyWavelets provides several methods of signal extrapolation that can be used to minimize
    edge effects. PyWavelet's default is 'sym':

    zpd - zero-padding - signal is extended by adding zero samples:
    ... 0  0 | x1 x2 ... xn | 0  0 ...

    cpd - constant-padding - border values are replicated:
    ... x1 x1 | x1 x2 ... xn | xn xn ...

    sym - symmetric-padding - signal is extended by mirroring samples:
    ... x2 x1 | x1 x2 ... xn | xn xn-1 ...

    ppd - periodic-padding - signal is treated as a periodic one:
    ... xn-1 xn | x1 x2 ... xn | x1 x2 ...

    sp1 - smooth-padding - signal is extended according to the first derivatives calculated on
    the edges (straight line)

    DWT performed for these extension modes is slightly redundant, but ensures perfect
    reconstruction. To receive the smallest possible number of coefficients, computations can
    be performed with the periodization mode:

    per - periodization - is like periodic-padding but gives the smallest possible number of
    decomposition coefficients. IDWT must be performed with the same mode.
    """
    import pywt

    data = np.atleast_2d(data)
    nt = data.shape[1]
    # reconstructed signals always seem to have an even number of points. If the number of
    # input data points is odd, trim last data point from reconstructed signal:
    isodd = nt % 2
    # filter data in place, iterate over channels in rows:
    nchans = len(data)
    for chani in range(nchans):
        # decompose the signal:
        cs = pywt.wavedec(data[chani], wname, mode=mode, level=maxlevel)
        # destroy the appropriate approximation coefficients to get highpass data:
        cs[0] = None
        # reconstruct the signal:
        recsignal = pywt.waverec(cs, wname, mode=mode)
        ntrec = len(recsignal)
        data[chani] = recsignal[:ntrec-isodd]
    
    return data

def updatenpyfilerows(fname, rows, arr):
    """Given a numpy formatted binary file (usually with .npy extension,
    but not necessarily), update 0-based rows (first dimension) of the
    array stored in the file from arr. Works for arrays of any rank >= 1"""
    assert len(arr) >= 1 # has at least 1 row
    f = open(fname, 'r+b') # open in read+write binary mode
    # get .npy format version:
    major, minor = np.lib.format.read_magic(f)
    assert (major == 1 and minor == 0)
    # read header to move file pointer to start of array in file
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    assert shape == arr.shape
    assert fortran_order == np.isfortran(arr)
    assert dtype == arr.dtype
    arroffset = f.tell()
    rowsize = arr[0].size * dtype.itemsize # nbytes per row
    # sort rows so that we move efficiently from start to end of file
    rows = sorted(rows) # rows might be a set, list, tuple, or array, convert to list
    # update rows in file
    for row in rows:
        f.seek(arroffset + row*rowsize) # seek from start of file, row is 0-based
        f.write(arr[row])
    f.close()

def unpickler_find_global_0_7_to_0_8(oldmod, oldcls):
    """Required for unpickling .sort version 0.7 files and upgrading them to version 0.8.
    Rename class names that changed between the two versions. Unfortunately, you can't check
    the .sort version number until after unpickling, so this has to be done for all .sort
    files during unpickling - it can't be done after unpickling"""
    old2new_streammod = {'core': 'stream'}
    old2new_streamcls = {'Stream': 'SurfStream',
                         'SimpleStream': 'SimpleStream',
                         'TrackStream': 'MultiStream'}
    try:
        newmod = old2new_streammod[oldmod]
        newcls = old2new_streamcls[oldcls]
    except KeyError: # no old to new conversion
        exec('import %s' % oldmod)
        return eval('%s.%s' % (oldmod, oldcls))
    print('Rename on unpickle: %s.%s -> %s.%s' % (oldmod, oldcls, newmod, newcls))
    exec('import %s' % newmod)
    return eval('%s.%s' % (newmod, newcls))
