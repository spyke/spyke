"""Main spyke window"""

from __future__ import division
from __future__ import print_function

import sys
if sys.version_info.major > 2:
    print
    print("WARNING!!!: You're running spyke in Python 3.x., currently unsupported.\n"
          "            spyke only saves and loads .sort files correctly in Python 2.7.x,\n"
          "            don't attempt to do so in Python 3.x!")
    input('Hit ENTER to continue...')

__authors__ = ['Martin Spacek', 'Reza Lotun']

# set working directory to path of this module instead of path of script that launched python,
# otherwise Qt4 has problems finding the spyke.ui file:
from . import __path__
import os
os.chdir(__path__[0])

import sys
import platform
import time
import datetime
import gc
try:
    import cPickle as pickle
except ImportError:
    import pickle
import random
from copy import copy
from struct import unpack
from collections import OrderedDict as odict

import numpy as np
import scipy.stats

# instantiate an IPython embedded shell which shows up in the terminal on demand
# and on every exception:
from IPython.terminal.ipapp import load_default_config
from IPython.terminal.embed import InteractiveShellEmbed
config = load_default_config()
# automatically call the pdb debugger after every exception, override default config:
config.TerminalInteractiveShell.pdb = True
ipshell = InteractiveShellEmbed(display_banner=False, config=config)

from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import Qt
getSaveFileName = QtGui.QFileDialog.getSaveFileName
getExistingDirectory = QtGui.QFileDialog.getExistingDirectory
SpykeUi, SpykeUiBase = uic.loadUiType('spyke.ui')

import pylab as pl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pyximport
pyximport.install(build_in_temp=False, inplace=True)
from . import util # .pyx file
from .gac import gac # .pyx file

from . import core
from .core import (toiter, tocontig, intround, intceil, printflush, lstrip, matlabize,
                   g, dist, iterable, ClusterChange, SpykeToolWindow, DJS,
                   qvar2list, qvar2str, merge_intervals)
from . import dat, nsx, surf, stream, probes
from .stream import SimpleStream, MultiStream
from .sort import Sort, SortWindow, NSLISTWIDTH, MEANWAVEMAXSAMPLES, NPCSPERCHAN
from .plot import SpikePanel, ChartPanel, LFPPanel
from .detect import Detector, calc_SPIKEDTYPE, DEBUG
from .extract import Extractor
from .cluster import Cluster, ClusterWindow
from .__version__ import __version__

# spike window temporal window (us)
SPIKETW = {'.dat': (-500, 1500),
           '.ns6': (-500, 1500),
           '.srf': (-400, 600),
           '.tsf': (-1000, 2000)}
# chart window temporal window (us)
CHARTTW = {'.dat': (-25000, 25000),
           '.ns6': (-25000, 25000),
           '.srf': (-25000, 25000),
           '.tsf': (-50000, 50000)}
# LFP window temporal window (us)
LFPTW = -500000, 500000

# zero out +/- this amount of time around each saturated timepoint when exporting
# high-pass data to KiloSort:
SATURATIONWINDOW = 500000 # us

# shift imported KiloSort spike times by this much for better positioning in sort window:
KILOSORTSHIFTCORRECT = -100 # us

# spatial channel layout:
# UVPERUM affects vertical channel spacing and voltage gain (which is further multiplied by
# each plot window's gain):
UVPERUM = {'.dat': 5, '.ns6': 5, '.srf': 2, '.tsf': 20}
# USPERUM affects horizontal channel spacing. Decreasing USPERUM increases horizontal overlap
# between spike chans. For .srf data, 17 gives roughly no horizontal overlap for
# self.tw[1] - self.tw[0] == 1000 us:
# However, this also depends on the horizontal spacing of the probe sites, so really
# this should be set according to probe type, not file type, or it should be scaled in
# terms of fraction of the horizontal span of the probe site layout:
USPERUM = {'.dat': 50, '.ns6': 50, '.srf': 17, '.tsf': 125}

DYNAMICNOISEX = {'.dat': 4.5, '.ns6': 4.5, '.srf': 6, '.tsf': 3} # noise multiplier
DT = {'.dat': 600, '.ns6': 600, '.srf': 400, '.tsf': 1500} # max time between spike peaks (us)

SCREENWIDTH = 1920 # TODO: this should be found programmatically
#SCREENHEIGHT = 1080 # TODO: this should be found programmatically
WINDOWTITLEHEIGHT = 26 # TODO: this should be found programmatically
BORDER = 2 # TODO: this should be found programmatically
SPIKEWINDOWWIDTHPERCOLUMN = 80
SPIKEWINDOWHEIGHT = 658 + 2*BORDER # TODO: this should be calculated from SCREENHEIGHT
CHARTWINDOWSIZE = 900+2*BORDER, SPIKEWINDOWHEIGHT
LFPWINDOWSIZE = 250+2*BORDER, SPIKEWINDOWHEIGHT
#SHELLSIZE = CHARTWINDOWSIZE[0], CHARTWINDOWSIZE[1]/2
CLUSTERWINDOWHEIGHT = 700

MAXRECENTFILES = 20 # anything > 10 will mess up keyboard accelerators, but who cares
WINDOWUPDATEORDER = ['Spike', 'LFP', 'Chart'] # chart goes last cuz it's slowest

# if updating at least this many selected spikes in .wave file, update them all
# instead for speed:
NDIRTYSIDSTHRESH = 200000


class SpykeWindow(QtGui.QMainWindow):
    """spyke's main window, uses gui layout generated by QtDesigner"""
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = SpykeUi()
        self.ui.setupUi(self) # lay it out
        self.groupMenuFiltering()
        self.groupMenuCAR()
        self.groupMenuSampling()
        self.addRecentFileActions()
        self.updateRecentFiles()
        
        self.move(0, 0) # top left corner, to make space for data windows

        self.streampath = os.getcwd() # init
        self.sortpath = os.getcwd() # init
        for d in ('~/data', '/data'): # use first existing of these paths, if any
            path = os.path.expanduser(d)
            if os.path.exists(path):
                self.streampath = path
                self.sortpath = path
                break
        self.windows = {} # holds child windows
        self.t = None # current time position in recording (us)

        self.hpstream = None
        self.lpstream = None

        self.cchanges = core.Stack() # cluster change stack, for undo/redo
        self.cci = -1 # pointer to cluster change for the next undo (add 1 for next redo)

        self.dirtysids = set() # sids whose waveforms in .wave file are out of date
        
        # disable most widgets until a stream or a sort is opened:
        self.EnableStreamWidgets(False)
        self.EnableSortWidgets(False)
        self.EnableFilteringMenu(False) # disable by default, not all file types need filtering
        self.EnableCARMenu(False) # disable until stream is open
        self.EnableSamplingMenu(False) # disable until stream is open
        
    def addRecentFileActions(self):
        """Init recent file QActions and insert them into the right place in the
        File menu. Leave them invisible until needed"""
        self.recentFileActions = []
        for i in range(MAXRECENTFILES):
            action = QtGui.QAction(self)
            action.setVisible(False)
            action.triggered.connect(self.OpenRecentFile)
            self.recentFileActions.append(action)
            self.ui.menuFile.insertAction(self.ui.actionSaveSort, action)
        self.ui.menuFile.insertSeparator(self.ui.actionSaveSort)

    def groupMenuFiltering(self):
        """Group filtering methods in filtering menu into a QActionGroup such that only
        one is ever active at a time. This isn't possible to do from within
        QtDesigner 4.7, so it's done here manually instead"""
        ui = self.ui
        filteringGroup = QtGui.QActionGroup(self)
        filteringGroup.addAction(ui.actionFiltmethNone)
        filteringGroup.addAction(ui.actionFiltmethBW)
        filteringGroup.addAction(ui.actionFiltmethBWNC)
        filteringGroup.addAction(ui.actionFiltmethWMLDR)

    def groupMenuCAR(self):
        """Group common average referencing methods in CAR menu into a QActionGroup such
        that only one is ever active at a time. This isn't possible to do from within
        QtDesigner 4.7, so it's done here manually instead"""
        ui = self.ui
        CARGroup = QtGui.QActionGroup(self)
        CARGroup.addAction(ui.actionCARNone)
        CARGroup.addAction(ui.actionCARMedian)
        CARGroup.addAction(ui.actionCARMean)

    def groupMenuSampling(self):
        """Group sampling rates in sampling menu into a QActionGroup such that only
        one is ever active at a time. This isn't possible to do from within
        QtDesigner 4.7, so it's done here manually instead"""
        ui = self.ui
        samplingGroup = QtGui.QActionGroup(self)
        samplingGroup.addAction(ui.action20kHz)
        samplingGroup.addAction(ui.action25kHz)
        samplingGroup.addAction(ui.action30kHz)
        samplingGroup.addAction(ui.action40kHz)
        samplingGroup.addAction(ui.action50kHz)
        samplingGroup.addAction(ui.action60kHz)
        samplingGroup.addAction(ui.action80kHz)
        samplingGroup.addAction(ui.action100kHz)
        samplingGroup.addAction(ui.action120kHz)

    @QtCore.pyqtSlot()
    def on_actionNewSort_triggered(self):
        self.DeleteSort() # don't create a new one until spikes exist

    @QtCore.pyqtSlot()
    def on_actionNewTrack_triggered(self):
        self.CreateNewTrack()

    def CreateNewTrack(self):
        """Create a new .track file"""
        exts = ['.ns6', '.dat', '.srf']
        caption = "Create .track file from %s files" % ' '.join(exts)
        starexts = [ '*%s' % ext for ext in exts ]
        filter = ('%s files ' % ', '.join(exts) +
                  '(%s)' % ' '.join(starexts) + ';;All files (*.*)')
        trackfname = getSaveFileName(self, caption=caption,
                                     directory=self.streampath,
                                     filter=filter)
        trackfname = str(trackfname)
        if not trackfname:
            return
        if not trackfname.endswith('.track'):
            trackfname += '.track'
        path = os.path.split(trackfname)[0]
        ls = os.listdir(path)
        fnames = {}
        for ext in exts:
            fnames = [ fname for fname in os.listdir(path) if fname.endswith(ext) ]
            if len(fnames) > 0:
                break
        if len(fnames) == 0:
            print("Couldn't find any .ns6, .dat, or .srf files in %r" % path)
            return
        fnames = sorted(fnames)
        trackstr = '\n'.join(fnames)
        with open(trackfname, 'w') as trackf:
            trackf.write(trackstr)
            trackf.write('\n') # end the file with a newline
        print('Wrote track file %r:' % trackfname)
        print(trackstr)
        self.OpenFile(trackfname)

    @QtCore.pyqtSlot()
    def on_actionOpen_triggered(self):
        getOpenFileName = QtGui.QFileDialog.getOpenFileName
        filter = (".dat, .ns6, .srf, .track, .tsf, .mat, .event & .sort files "
                  "(*.dat *.ns6 *.srf *.track *.tsf *.mat *.event*.zip *.sort );;"
                  "All files (*.*)")
        fname = getOpenFileName(self, caption="Open stream or sort",
                                directory=self.streampath,
                                filter=filter)
        fname = str(fname)
        if fname:
            self.OpenFile(fname)

    @QtCore.pyqtSlot()
    def on_actionSaveSort_triggered(self):
        try:
            self.sort
        except AttributeError: # sort doesn't exist
            return
        if self.sort.fname:
            self.SaveSortFile(self.sort.fname) # save to existing sort fname
        else:
            self.on_actionSaveSortAs_triggered()

    @QtCore.pyqtSlot()
    def on_actionSaveSortAs_triggered(self):
        """Save sort to new .sort file"""
        defaultfname = os.path.join(self.sortpath, self.sort.fname)
        if self.sort.fname == '': # sort hasn't been previously saved
            # generate default fname with hpstream.fname:
            fname = self.hpstream.fname.replace(' ', '_')
            # and datetime:
            #dt = str(datetime.datetime.now()) # get a sort creation timestamp
            #dt = dt.split('.')[0] # ditch the us
            #dt = dt.replace(' ', '_')
            #dt = dt.replace(':', '.')
            #defaultfname += fname + '_' + dt + '.sort'
            defaultfname += fname + '.sort'
        fname = getSaveFileName(self, caption="Save sort As",
                                directory=defaultfname,
                                filter="Sort files (*.sort);;"
                                       "All files (*.*)")
        fname = str(fname)
        if fname:
            base, ext = os.path.splitext(fname)
            if ext != '.sort':
                fname = base + '.sort' # make sure it has .sort extension
            head, tail = os.path.split(fname)
            self.sortpath = head # update sort path
            # make way for new .spike and .wave files
            try: del self.sort.spikefname
            except AttributeError: pass
            try: del self.sort.wavefname
            except AttributeError: pass
            self.SaveSortFile(tail)

    @QtCore.pyqtSlot()
    def on_actionSaveTrackChans_triggered(self):
        self.SaveTrackChans()

    def SaveTrackChans(self):
        """Overwrite existing .track file, potentially saving a new set of enabled chans"""
        stream = self.hpstream
        if not stream.is_multi():
            print("Stream is not a MultiStream, can't save a .track file")
            return
        trackfname = os.path.join(self.streampath, stream.fname)
        if not os.path.isfile(trackfname):
            raise RuntimeError('somehow the current MultiStream has no existing .track file')
        trackstr = ''
        allchans = np.sort(stream.streams[0].f.fileheader.chans)
        if len(stream.chans) != len(allchans):
            # some chans are disabled, write them as a comment in .track file
            trackstr += '# enabledchans = %r\n' % list(stream.chans)
        else:
            assert (stream.chans == allchans).all()
        trackstr += '\n'.join(stream.fnames)
        with open(trackfname, 'w') as trackf:
            trackf.write(trackstr)
            trackf.write('\n') # end the file with a newline
        print('Wrote track file %r:' % trackfname)
        print(trackstr)

    @QtCore.pyqtSlot()
    def on_actionSaveParse_triggered(self):
        if self.hpstream.ext == '.srf':
            self.hpstream.pickle()
        else:
            print('Only .srf streams have complicated parsings that can be '
                  'saved to a .parse file')

    @QtCore.pyqtSlot()
    def on_actionExportPtcsFiles_triggered(self):
        path = getExistingDirectory(self, caption="Export .ptcs file(s) to",
                                    directory=self.sortpath)
        path = str(path)
        if path:
            self.sort.exportptcsfiles(path, self.sortpath)
            # don't update path

    @QtCore.pyqtSlot()
    def on_actionExportTsChIdFiles_triggered(self):
        path = getExistingDirectory(self, caption="Export .tschid file(s) to",
                                    directory=self.sortpath)
        path = str(path)
        if path:
            self.sort.exporttschid(path)
            # don't update path

    @QtCore.pyqtSlot()
    def on_actionExportDIN_triggered(self):
        path = getExistingDirectory(self, caption="Export .din file(s) to",
                                    directory=self.sortpath)
        path = str(path)
        if path:
            ## TODO: if sort doesn't exist, make a temporary fake with hpstream
            ## as its stream. That's all that's needed.
            self.sort.exportdin(path)
            # don't update path

    @QtCore.pyqtSlot()
    def on_actionExportTextheader_triggered(self):
        path = getExistingDirectory(self, caption="Export .textheader file(s) to",
                                    directory=self.sortpath)
        path = str(path)
        if path:
            ## TODO: if sort doesn't exist, make a temporary fake with hpstream
            ## as its stream. That's all that's needed.
            self.sort.exporttextheader(path)
            # don't update path

    @QtCore.pyqtSlot()
    def on_actionExportAll_triggered(self):
        path = getExistingDirectory(self,
                                    caption="Export .ptcs, .din and .textheader file(s) to",
                                    directory=self.sortpath)
        path = str(path)
        if path:
            self.sort.exportall(basepath=path, sortpath=self.sortpath)
            # don't update path

    @QtCore.pyqtSlot()
    def on_actionExportCSVFile_triggered(self):
        """Export "good" spikes to .csv file"""
        sortfname = os.path.join(self.sortpath, self.sort.fname)
        if sortfname == '': # sort hasn't been previously saved
            raise ValueError('Please save .sort file before exporting to .csv')
        # generate default fname with sort fname + datetime:
        sortfname = sortfname.replace(' ', '_')
        dt = str(datetime.datetime.now()) # get an export timestamp
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        ext = '.csv'
        defaultfname = sortfname + '_' + dt + ext
        caption = "Export spikes to %s file" % ext
        filter = "%s spike files (*%s);;All files (*.*)" % (ext, ext)
        fname = getSaveFileName(self, caption=caption,
                                directory=defaultfname,
                                filter=filter)
        fname = str(fname)
        if fname:
            before, sep, after = fname.partition(ext)
            if sep != ext:
                fname = before + ext # make sure it has extension
            sw = self.OpenWindow('Sort') # in case it isn't already open
            self.sort.exportcsv(fname)

    @QtCore.pyqtSlot()
    def on_actionExportSpikesZipFile_triggered(self):
        """Save selected spikes on selected channels and timepoints to
        binary .spikes.zip file"""
        self.exportSpikeWaveforms(format='binary')

    @QtCore.pyqtSlot()
    def on_actionExportSpikesCSVFile_triggered(self):
        """Save selected spikes on selected channels and timepoints to
        text .spikes.csv file"""
        self.exportSpikeWaveforms(format='text')

    def exportSpikeWaveforms(self, format):
        """Save selected spikes on selected channels and timepoints to
        binary .spikes.zip file or text .spikes.csv file"""
        if format == 'binary':
            ext = '.spikes.zip'
        elif format == 'text':
            ext = '.spikes.csv'
        else:
            raise ValueError("invalid format: %r" % format)
        defaultfname = os.path.join(self.sortpath, self.sort.fname)
        if defaultfname == '': # sort hasn't been previously saved
            # generate default fname with hpstream.fname and datetime
            fname = self.hpstream.fname.replace(' ', '_')
            dt = str(datetime.datetime.now()) # get an export timestamp
            dt = dt.split('.')[0] # ditch the us
            dt = dt.replace(' ', '_')
            dt = dt.replace(':', '.')
            defaultfname = fname + '_' + dt
        defaultfname = defaultfname + ext
        caption = "Export spike waveforms to %s %s file" % (format, ext)
        filter = "%s spike waveform files (*%s);;All files (*.*)" % (format, ext)
        fname = getSaveFileName(self, caption=caption,
                                directory=defaultfname,
                                filter=filter)
        fname = str(fname)
        if fname:
            before, sep, after = fname.partition(ext)
            if sep != ext:
                fname = before + ext # make sure it has extension
            sids = self.GetAllSpikes()
            selchans = self.get_selchans(sids)
            sw = self.OpenWindow('Sort') # in case it isn't already open
            tis = sw.tis
            self.sort.exportspikewaves(sids, selchans, tis, fname, format)

    @QtCore.pyqtSlot()
    def on_actionExportHighPassDatFiles_triggered(self):
        self.export_hpstream()

    def export_hpstream(self, cat=False, checksat=False, satwin=None,
                        export_msg='high-pass', export_ext='.filt.dat'):
        """Export high-pass stream to user-designated path, using current preprocessing
        settings (filtering, CAR, and resampling) and channel selection, to export_ext file(s)
        with associated export_ext.json file describing the preprocessing that was done. This
        can also be used to export raw data if the hpstream settings for filtering, CAR and
        resampling are set appropriately. Use export_msg and export_ext to communicate this.
        cat controls whether to concatenate all the exported data into a single
        .dat file. If checksat is true, check for saturation in raw data, then zero out +/-
        satwin us around any saturated data. This works best if the data is indeed high-pass"""
        if not self.hpstream:
            print('First open a stream!')
            return
        if self.hpstream.is_multi(): # self.hpstream is a MultiStream
            hpstreams = self.hpstream.streams
            defaultpath = hpstreams[0].f.path # get path of first stream
            if cat: # export entire MultiStream to one file:
                hpstreams = [self.hpstream]
        else: # self.hpstream is a single Stream
            hpstreams = [self.hpstream]
            defaultpath = hpstreams[0].f.path
            assert cat == False # nonsensical for a single Stream
        caption = "Export %s data to %s files" % (export_msg, export_ext)
        path = str(getExistingDirectory(self, caption=caption, directory=defaultpath))
        if not path:
            return

        print('Exporting %d channels:' % self.hpstream.nchans)
        print('chans = %s' % self.hpstream.chans)
        blocksize = int(float(self.ui.blockSizeLineEdit.text()))
        print('Exporting in blocks of %d us' % blocksize)
        for hps in hpstreams:
            fname = hps.fname + export_ext
            fullfname = os.path.join(path, fname)
            fulljsonfname = fullfname + '.json'
            print('Exporting %s data to %r' % (export_msg, fullfname))
            with open(fullfname, 'wb') as datf:
                ztrangess = []
                t0s = np.arange(hps.t0, hps.t1, blocksize)
                for t0 in t0s:
                    t1 = t0 + blocksize
                    #print('%d to %d us' % (t0, t1))
                    printflush('.', end='') # succint progress indicator
                    wave = hps(t0, t1, checksat=checksat)
                    data = wave.data
                    if checksat:
                        satis = wave.satis # should have same shape as data
                        if satis.any():
                            nt = data.shape[1] # num timepoints in this block
                            wsatis = np.where(satis) # integer row and col indices
                            satchanis = np.unique(wsatis[0]) # indices of rows that saturated
                            satchans = wave.chans[satchanis]
                            print() # newline
                            print('Saturation in block (%d, %d) on chans %s'
                                  % (t0, t1, satchans))
                            ntwin = intround(satwin / hps.tres)
                            sattis = satis.any(axis=0) # time only, collapse across all chans
                            edges = np.diff(sattis.astype(int)) # find +ve and -ve edges
                            onis = np.where(edges > 0)[0] + 1
                            offis = np.where(edges < 0)[0] + 1
                            if len(onis) - len(offis) == 1:
                                offis = np.append(offis, nt) # last off is end of block
                            elif len(offis) - len(onis) == 1:
                                onis = np.append(onis, 0) # first on is start of block
                            # convert to nx2 array, expand window for zeroing around on
                            # and off index of each saturation:
                            ztrangeis = np.stack([onis-ntwin, offis+ntwin], axis=1)
                            ztrangeis = np.asarray(merge_intervals(ztrangeis)) # remove overlap
                            ztrangeis = ztrangeis.clip(0, nt) # limit to valid slice values
                            for oni, offi in ztrangeis:
                                sattis[oni:offi] = True
                            data[:, sattis] = 0 # zero out data at sattis
                            ztrangeis = ztrangeis.clip(max=nt-1) # limit to valid index values
                            ztranges = wave.ts[ztrangeis]
                            print('Zeroed-out time ranges:')
                            print(intround(ztranges)) # convert to int for better display
                            ztrangess.append(ztranges)
                    #if t0 == t0s[-1]:
                    #    print('last block asked:', t0, t1)
                    #    print('last block received:', wave.ts[0], wave.ts[-1])
                    data.T.tofile(datf) # write in column-major (Fortran) order
                print() # newline
                core.write_dat_json(hps, fulljsonfname)
                if ztrangess:
                    ztrangess = np.concatenate(ztrangess, axis=0)
                    ztrangesfname = fullfname + '.0tranges.npy'
                    np.save(ztrangesfname, ztrangess)
                    print('Zeroed-out %d time ranges' % len(ztrangess))
                    print('Wrote 0tranges file %r' % ztrangesfname)
        print('Done exporting %s data' % export_msg)

        # only return path and fname if we're only exporting to a single file:
        if len(hpstreams) == 1:
            return path, fname

    @QtCore.pyqtSlot()
    def on_actionExportLFPZipFiles_triggered(self):
        self.export_lpstream(format='binary')

    @QtCore.pyqtSlot()
    def on_actionExportLFPCSVFiles_triggered(self):
        self.export_lpstream(format='text')

    def export_lpstream(self, format='binary'):
        """Export low-pass stream (LFP) data as binary .lfp.zip file(s) or text .lfp.csv
        file(s) in user-designated basepath"""
        if not self.lpstream:
            print('First open a stream!')
            return
        format2ext = {'binary': '.lfp.zip', 'text': '.lfp.csv'}
        ext = format2ext[format]
        caption = "Export low-pass data to %s %s files" % (format, ext)
        basepath = getExistingDirectory(self, caption=caption, directory=self.sortpath)
        basepath = str(basepath)
        if not basepath:
            return
        if self.lpstream.is_multi(): # self.lpstream is a MultiStream
            lpstreams = self.lpstream.streams
        else: # self.lpstream is a single Stream
            lpstreams = [self.lpstream]
        print('Exporting low-pass data to:')
        for lps in lpstreams:
            path = os.path.join(basepath, lps.srcfnameroot)
            try: os.mkdir(path)
            except OSError: pass # path already exists?
            fullfname = os.path.join(path, lps.srcfnameroot+ext)
            print(fullfname)
            # collect low-pass data in blocks, to prevent MemoryErrors when trying to
            # low-pass filter an entire raw ephys data file:
            blocksize = int(float(self.ui.blockSizeLineEdit.text())) # allow exp notation
            t0s = np.arange(lps.t0, lps.t1, blocksize)
            data = []
            for t0 in t0s:
                t1 = t0 + blocksize
                wave = lps[t0:t1]
                data.append(wave.data)
            # concatenate data blocks horizontally in time:
            data = np.hstack(data)
            if format == 'binary':
                chanpos = lps.probe.siteloc_arr()
                uVperAD = lps.converter.AD2uV(1)
                with open(fullfname, 'wb') as f:
                    np.savez_compressed(f, data=data, chans=wave.chans, t0=lps.t0,
                                        t1=lps.t1, tres=lps.tres, chanpos=chanpos,
                                        chan0=lps.probe.chan0, probename=lps.probe.name,
                                        uVperAD=uVperAD)
            else: # format == 'text'
                np.savetxt(fullfname, data, fmt='%d', delimiter=',') # data should be int
        print('Done exporting low-pass data')

    @QtCore.pyqtSlot()
    def on_actionExportHighPassEnvelopeDatFiles_triggered(self):
        self.export_hp_envelope()

    @QtCore.pyqtSlot()
    def on_actionExportHighPassBipolarRefEnvelopeDatFiles_triggered(self):
        self.export_hp_envelope(bipolarref=True)

    def export_hp_envelope(self, sampfreq=2000, f0=None, f1=500, bipolarref=False):
        """Export envelope of high-pass stream to user-designated path, using current
        preprocessing settings (filtering, CAR, and resampling), to .envl.dat file(s) with
        associated .envl.dat.json file describing the preprocessing that was done. Decimate
        output to get sampfreq. Export chans in order of depth, superficial to deep.
        bipolarref: optionally take each channel's raw data to be the difference of the two
        immediately spatially adjacent channels, before calculating the envelope"""
        if not self.hpstream:
            print('First open a stream!')
            return
        if self.hpstream.is_multi(): # self.hpstream is a MultiStream
            hpstreams = self.hpstream.streams
        else: # self.hpstream is a single Stream
            hpstreams = [self.hpstream]
        defaultpath = hpstreams[0].f.path
        caption = "Export envelope of high-pass, preprocessed data to .envl.dat files"
        path = str(getExistingDirectory(self, caption=caption, directory=defaultpath))
        if not path:
            return
        print('Exporting high-pass envelope data to:')
        for hps in hpstreams:
            assert hps.sampfreq % sampfreq == 0
            decimatex = intround(hps.sampfreq / sampfreq)
            fullfname = os.path.join(path, hps.fname + '.envl.dat')
            fulljsonfname = fullfname + '.json'
            print(fullfname)
            # excess data to get at either end of each block, to eliminate
            # filtering edge effects:
            xs = core.XSWIDEBANDPOINTS * hps.rawtres # us
            # sort channels for export by depth instead of by ID:
            # get ypos of each enabled site:
            enabledchans = self.hpstream.chans
            ypos = [ self.hpstream.probe.SiteLoc[chan][1] for chan in enabledchans ]
            ysortis = np.argsort(ypos)
            ychans = list(enabledchans[ysortis])
            with open(fullfname, 'wb') as datf:
                blocksize = int(float(self.ui.blockSizeLineEdit.text())) # allow exp notation
                t0s = np.arange(hps.t0, hps.t1, blocksize)
                for t0 in t0s:
                    t1 = t0 + blocksize
                    t0xs, t1xs = t0-xs, t1+xs
                    wave = hps[t0xs:t1xs] # get excess range of data
                    data = wave.data[ysortis] # sort chans by depth
                    chans = wave.chans[ysortis]
                    assert list(chans) == ychans
                    if bipolarref:
                        # set each channel to be the difference of the two immediately
                        # spatially adjacent channels:
                        data[1:-1] = data[:-2] - data[2:]
                        data[[0, -1]] = 0 # null out the first and last channel
                    # get envelope of data by rectifying and low-pass filtering:
                    data = core.envelope_filt(data, sampfreq=hps.sampfreq,
                                              f0=f0, f1=f1) # float64
                    # ensure data limits fall within int16:
                    iint16 = np.iinfo(np.int16)
                    assert data.max() <= iint16.max
                    assert data.min() >= iint16.min
                    data = np.int16(data) # convert float64 to int16
                    t0i, t1i = wave.ts.searchsorted([t0, t1]) # get indices to remove excess
                    data = data[:, t0i:t1i:decimatex] # remove excess and decimate
                    data.T.tofile(datf) # write in column-major (Fortran) order
                envelope = odict()
                envelope['meth'] = 'abs'
                envelope['bipolar_ref'] = bipolarref
                envelope['filter_meth'] = 'BW'
                envelope['f0'] = f0
                envelope['f1'] = f1
                core.write_dat_json(hps, fulljsonfname, sampfreq=sampfreq,
                                    chans=ychans, chan_order='depth', envelope=envelope)
        print('Done exporting high-pass envelope data')

    @QtCore.pyqtSlot()
    def on_actionExportHighPassDatKiloSortFiles_triggered(self):
        self.export_hp_ks_dat()

    @QtCore.pyqtSlot()
    def on_actionExportRawDataDatFiles_triggered(self):
        self.export_raw_dat()

    def export_hp_ks_dat(self):
        """Export high-pass ephys data for use in KiloSort, while checking
        for and zeroing out any periods of saturation. Exports enabled chans concatenated
        across all files in current track, to .dat file in user-designated path.
        This works by first turning off all filtering, CAR, and resampling, then calling
        self.export_hpstream(), then restoring filtering, CAR, and resampling settings"""
        print('Exporting high-pass ephys data to .dat file for use in KiloSort, '
              'removing any saturation')

        # save current hpstream filtering CAR and sampling settings:
        stream = self.hpstream
        if not stream:
            print('First open a stream!')
            return
        filtmeth = stream.filtmeth
        car = stream.car
        sampfreq = stream.sampfreq
        shcorrect = stream.shcorrect

        # set hpstream to show high-pass filtered, but otherwise raw, data:
        print('Temporarily setting filtering to non-causal Butterworth, '
              'disabling CAR & resampling')
        self.SetFiltmeth('BWNC')
        self.SetCAR(None)
        self.SetSampfreq(stream.rawsampfreq)
        if stream.ext != '.srf':
            self.SetSHCorrect(False) # leave it enabled for .srf, data is wrong w/o it

        # do the export:
        if stream.is_multi(): # it's a MultiStream
            cat = True # concatenate
        else: # it's a single Stream
            cat = False # nothing to concatenate
        result = self.export_hpstream(cat=cat, checksat=True, satwin=SATURATIONWINDOW,
                                      export_msg='high-pass', export_ext='.dat')
        if result:
            path, datfname = result

        # restore hpstream settings:
        print('Restoring filtering, CAR, and resampling settings')
        self.SetFiltmeth(filtmeth)
        self.SetCAR(car)
        self.SetSampfreq(sampfreq)
        self.SetSHCorrect(shcorrect)

        if not result:
            print('Raw data export cancelled')
            return

        # write KiloSort channel map .mat file, indicate which chans are included in the .dat
        datfnameML = matlabize(datfname) # make suitable for use as MATLAB script name
        chanmapfname = datfnameML + '_ks_chanmap.mat'
        fullchanmapfname = os.path.join(path, chanmapfname)
        core.write_ks_chanmap_mat(stream, fullchanmapfname)

        # write KiloSort config .m file:
        with open('./templates/kilosort/ks_config.m') as templateksconfigf:
            ksconfigstr = templateksconfigf.read()
        # nclusts for KiloSort to use: 3x nchans, rounded up to nearest multiple of 32:
        nclusts = intceil(3 * stream.nchans / 32) * 32
        ksconfigstr = ksconfigstr.format(DATFNAME=datfname,
                                         KSRESULTSFNAME=datfname + '.ks_results',
                                         FS=stream.rawsampfreq,
                                         NCHANS=stream.nchans,
                                         NCLUSTS=nclusts,
                                         CHANMAPFNAME=chanmapfname)
        ksconfigfname = datfnameML + '_ks_config.m'
        fullksconfigfname = os.path.join(path, ksconfigfname)
        with open(fullksconfigfname, 'w') as ksconfigf:
            ksconfigf.write(ksconfigstr)
        print('Wrote KiloSort config file %r' % fullksconfigfname)

        # write KiloSort run .m file:
        with open('./templates/kilosort/ks_run.m') as templateksrunf:
            ksrunstr = templateksrunf.read()
        ksrunstr = ksrunstr.format(KSCONFIGFNAME=ksconfigfname)
        ksrunfname = datfnameML + '_ks_run.m'
        fullksrunfname = os.path.join(path, ksrunfname)
        with open(fullksrunfname, 'w') as ksrunf:
            ksrunf.write(ksrunstr)
        print('Wrote KiloSort run file %r' % fullksrunfname)

    def export_raw_dat(self):
        """Export raw ephys data of enabled chans concatenated across all files in current
        track, to .dat file in user-designated path. This works by first turning off all
        filtering, CAR, and resampling, then calling self.export_hpstream(), then restoring
        filtering, CAR, and resampling settings"""
        print('Exporting raw ephys data to .dat file')

        # save current hpstream filtering CAR and sampling settings:
        stream = self.hpstream
        if not stream:
            print('First open a stream!')
            return
        filtmeth = stream.filtmeth
        car = stream.car
        sampfreq = stream.sampfreq
        shcorrect = stream.shcorrect

        # set hpstream to show raw data:
        print('Temporarily disabling filtering, CAR, and resampling for raw export')
        self.SetFiltmeth(None)
        self.SetCAR(None)
        self.SetSampfreq(stream.rawsampfreq)
        if stream.ext != '.srf':
            self.SetSHCorrect(False) # leave it enabled for .srf, data is wrong w/o it

        # do the export:
        if stream.is_multi(): # it's a MultiStream
            cat = True # concatenate
        else: # it's a single Stream
            cat = False # nothing to concatenate
        result = self.export_hpstream(cat=cat, export_msg='raw', export_ext='.dat')
        if result:
            path, datfname = result

        # restore hpstream settings:
        print('Restoring filtering, CAR, and resampling settings')
        self.SetFiltmeth(filtmeth)
        self.SetCAR(car)
        self.SetSampfreq(sampfreq)
        self.SetSHCorrect(shcorrect)

        if not result:
            print('Raw data export cancelled')
            return

    @QtCore.pyqtSlot()
    def on_actionConvertKiloSortNpy2EventsZip_triggered(self):
        caption = "Convert relevant KiloSort .npy files to a single .events.zip file"
        path = getExistingDirectory(self, caption=caption, directory=self.streampath)
        path = str(path)
        if not path:
            return
        self.convert_kilosortnpy2eventszip(path)

    def update_sort_version(self):
        """Update self.sort to latest version"""
        s = self.sort
        v = float(s.__version__) # sort version
        lv = float(__version__) # latest version
        if v > lv:
            raise RuntimeError('versioning error')
        if v == lv:
            print('No update necessary')
            return
        if v < 0.3:
            print("Can't auto update from sort version < 0.3")
            return
        if v == 0.3:
            v = self.update_0_3_to_0_4()
        if v == 0.4:
            v = self.update_0_4_to_0_5()
        if v == 0.5:
            v = self.update_0_5_to_0_6()
        if v == 0.6:
            v = self.update_0_6_to_0_7()
        if v == 0.7:
            v = self.update_0_7_to_0_8()
        if v == 0.8:
            v = self.update_0_8_to_0_9()
        if v == 0.9:
            v = self.update_0_9_to_1_0()
        if v == 1.0:
            v = self.update_1_0_to_1_1()
        if v == 1.1:
            v = self.update_1_1_to_1_2()
        if v == 1.2:
            v = self.update_1_2_to_1_3()
        print('Now save me!')
            
    def update_0_3_to_0_4(self):
        """Update sort 0.3 to 0.4:
            - reload all spike waveforms and fix all of their time values
        """        
        print('Updating sort from version 0.3 to 0.4')
        s = self.sort
        sids = np.arange(s.nspikes)
        s.reload_spikes(sids)
        # add sids to the set of dirtysids to be resaved to .wave file:
        self.dirtysids.update(sids)
        s.__version__ = '0.4' # update
        print('Done updating sort from version 0.3 to 0.4')
        return float(s.__version__)
        
    def update_0_4_to_0_5(self):
        """Update sort 0.4 to 0.5:
            - rename sort.sortfname to sort.fname
        """
        print('Updating sort from version 0.4 to 0.5')
        s = self.sort
        s.fname = s.sortfname
        del s.sortfname
        s.__version__ = '0.5' # update
        print('Done updating sort from version 0.4 to 0.5')
        return float(s.__version__)

    def update_0_5_to_0_6(self):
        """Update sort 0.5 to 0.6:
            - rename sort.spikes field names 'phasetis' and 'dphase' to
              'tis' and 'dt' respectively
            - remove unused 'cid', 's0' and 's1' fields from sort.spikes, reorder fields
        """
        print('Updating sort from version 0.5 to 0.6')
        s = self.sort
        names = list(s.spikes.dtype.names) # convert from tuple
        phasetis_index = names.index('phasetis')
        dphase_index = names.index('dphase')
        assert (phasetis_index, dphase_index) == (13, 19)
        names[phasetis_index] = 'tis' # rename 'phasetis' to 'tis'
        names[dphase_index] = 'dt' # rename 'dphase' to 'dt'
        s.spikes.dtype.names = names # checks length and auto converts back to tuple
        # also rename fields in detector's SPIKEDTYPE:
        for i in [phasetis_index, dphase_index]:
            field = list(s.detector.SPIKEDTYPE[i])
            field[0] = names[i]
            s.detector.SPIKEDTYPE[i] = tuple(field)

        # new name order, leaves out unused 'cid', 's0' and 's1'
        newnames = ['id', 'nid', 'chan', 'nchans', 'chans', 'chani', 't', 't0', 't1', 'dt',
                    'tis', 'aligni', 'V0', 'V1', 'Vpp', 'x0', 'y0', 'sx', 'sy']
        olddtype = s.detector.SPIKEDTYPE # list of tuples
        oldnames = [ field[0] for field in olddtype ]
        newdtype = []
        for name in newnames:
            newdtype.append(olddtype[oldnames.index(name)])
        s.detector.SPIKEDTYPE = newdtype # replace detector's SPIKEDTYPE
        newspikes = np.empty(s.spikes.shape, dtype=newdtype)
        from numpy.lib import recfunctions as rfn
        newspikes = rfn.recursive_fill_fields(s.spikes, newspikes) # copy from old to new
        s.spikes = newspikes # overwrite

        # in cluster.pos and .normpos, remove 's0' and 's1', and rename 'dphase' to 'dt':
        for c in s.clusters.values():
            c.pos.pop('s0')
            c.pos.pop('s1')
            c.pos['dt'] = c.pos.pop('dphase')
            c.normpos.pop('s0')
            c.normpos.pop('s1')
            c.normpos['dt'] = c.normpos.pop('dphase')

        s.__version__ = '0.6' # update
        print('Done updating sort from version 0.5 to 0.6')
        return float(s.__version__)

    def update_0_6_to_0_7(self):
        """Update sort 0.6 to 0.7:
            - replace sort.TW class attribute with sort.tw instance attribute
        """
        print('Updating sort from version 0.6 to 0.7')
        s = self.sort
        # Sort.TW class attrib was (-500, 500) in version 0.6
        s.tw = -500, 500
        s.__version__ = '0.7' # update
        print('Done updating sort from version 0.6 to 0.7')
        return float(s.__version__)

    def update_0_7_to_0_8(self):
        """Update sort 0.7 to 0.8:
            - rename/move classes (done by core.unpickler_find_global()):
                - core.Stream -> stream.SurfStream
                - core.SimpleStream -> stream.SimpleStream
                - core.TrackStream -> stream.MultiStream
            - rename Stream attrib .srff -> .f
            - rename MultiStream attrib .srffnames -> .fnames
            - add potentially missing sort.npcsperchan attrib
        """
        print('Updating sort from version 0.7 to 0.8')
        s = self.sort
        stream = s.stream
        classname = stream.__class__.__name__
        if classname == 'SurfStream':
            f = stream.srff
            del stream.srff
            stream.f = f
        elif classname == 'SimpleStream':
            # don't think any existing saved SimpleStreams had a .srff attrib:
            pass
        elif classname == 'MultiStream':
            fnames = stream.srffnames
            del stream.srffnames
            stream.fnames = fnames
        else:
            raise RuntimeError("don't know how to upgrade stream type %r" % classname)
        try:
            s.npcsperchan
        except AttributeError:
            s.npcsperchan = NPCSPERCHAN

        s.__version__ = '0.8' # update
        print('Done updating sort from version 0.7 to 0.8')
        return float(s.__version__)

    def update_0_8_to_0_9(self):
        """Update sort 0.8 to 0.9:
            - add sort.filtmeth attrib, init to None
        """
        print('Updating sort from version 0.8 to 0.9')
        s = self.sort
        try:
            s.filtmeth
        except AttributeError:
            s.filtmeth = None
        s.__version__ = '0.9' # update
        print('Done updating sort from version 0.8 to 0.9')
        return float(s.__version__)

    def update_0_9_to_1_0(self):
        """Update sort 0.9 to 1.0:
            - add nlockchans and lockchans fields to spike record
            - add detector.lockrx attrib
        """
        print('Updating sort from version 0.9 to 1.0')
        s = self.sort
        oldspikes = s.spikes

        olddtype = oldspikes.dtype.descr # [(fieldname, fieldtype)] tuples, ordered by offset
        oldnames = oldspikes.dtype.names # list of field names, ordered by offset
        oldfields = oldspikes.dtype.fields # {fieldname:(fielddtype, byte offset)} mapping

        newdtype = copy(olddtype)
        inserti = oldnames.index('t') # insert our new fields just before the 't' field
        assert inserti == 6
        newdtype.insert(inserti, ('nlockchans', oldfields['nchans'][0])) # copy nchans type
        newdtype.insert(inserti+1, ('lockchans', oldfields['chans'][0])) # copy chans type

        s.detector.SPIKEDTYPE = newdtype # replace detector's SPIKEDTYPE
        newspikes = np.empty(oldspikes.shape, dtype=newdtype) # init newspikes
        from numpy.lib import recfunctions as rfn
        newspikes = rfn.recursive_fill_fields(oldspikes, newspikes) # copy from old to new
        # the new fields are redundant for old detection runs, but are used in the code
        # for displaying spike rasters:
        newspikes['nlockchans'] = oldspikes['nchans']
        newspikes['lockchans'] = oldspikes['chans']
        s.spikes = newspikes # overwrite

        from pprint import pprint
        print('old dtype:')
        pprint(olddtype)
        print('new dtype:')
        pprint(s.spikes.dtype.descr)

        # add new detector.lockrx attrib, supercedes detector.lockr attrib
        s.detector.lockrx = 0.0 # set to 0 to indicate it wasn't used during detection

        s.__version__ = '1.0' # update
        print('Done updating sort from version 0.9 to 1.0')
        return float(s.__version__)

    def update_1_0_to_1_1(self):
        """Update sort 1.0 to 1.1:
            - add sort.car attrib, init to None
        """
        print('Updating sort from version 1.0 to 1.1')
        s = self.sort
        try:
            s.car
        except AttributeError:
            s.car = None
        s.__version__ = '1.1' # update
        print('Done updating sort from version 1.0 to 1.1')
        return float(s.__version__)

    def update_1_1_to_1_2(self):
        """Update sort 1.1 to 1.2:
            - add stream.adapter, fileheader.adapter & fileheader.adaptername, init to None
        """
        print('Updating sort from version 1.1 to 1.2')
        s = self.sort
        if s.stream.is_multi():
            s.stream.adapter = None
            streams = s.stream.streams
        else: # it's a single stream
            streams = [s.stream]
        for stream in streams: # iterate over all single streams
            stream.adapter = None
            if stream.ext in ['.ns6', '.dat']:
                stream.f.fileheader.adapter = None
                stream.f.fileheader.adaptername = None
        s.__version__ = '1.2' # update
        print('Done updating sort from version 1.1 to 1.2')
        return float(s.__version__)

    def update_1_2_to_1_3(self):
        """Update sort 1.2 to 1.3:
            - rename class (done by core.unpickler_find_global()):
                - A1x64_Poly2_6mm_23s_160 -> A1x64
        """
        print('Updating sort from version 1.2 to 1.3')
        s = self.sort
        classname = s.probe.__class__.__name__
        if s.probe.name == 'A1x64_Poly2_6mm_23s_160':
            print('sort.probe class is now %r' % classname)
            print('sort.probe.name was %r' % s.probe.name)
            s.probe.name = 'A1x64' # update name attribute
            print('sort.probe.name is now %r' % s.probe.name)
        s.__version__ = '1.3' # update
        print('Done updating sort from version 1.2 to 1.3')
        return float(s.__version__)

    @QtCore.pyqtSlot()
    def on_actionCloseSort_triggered(self):
        # TODO: add confirmation dialog if Sort not saved
        self.CloseSortFile()
        print('Closed sort')

    @QtCore.pyqtSlot()
    def on_actionCloseStream_triggered(self):
        if self.hpstream is not None:
            self.CloseStream()
            print('Closed stream')

    @QtCore.pyqtSlot()
    def on_actionQuit_triggered(self):
        self.close()
        #self.destroy() # no longer seems necessary, causes segfault

    def closeEvent(self, event):
        self.on_actionCloseSort_triggered()
        self.on_actionCloseStream_triggered()
        QtGui.QMainWindow.closeEvent(self, event)

    def keyPressEvent(self, event):
        key = event.key()
        try:
            sw = self.windows['Sort']
        except KeyError:
            QtGui.QMainWindow.keyPressEvent(self, event) # pass it on
        if key == Qt.Key_A:
            self.ui.plotButton.click()
        elif key == Qt.Key_N:
            self.ui.normButton.click()
        elif key in [Qt.Key_Escape, Qt.Key_E]:
            sw.clear()
        elif key == Qt.Key_R: # doesn't fire when certain widgets have focus
            sw.on_actionSelectRandomSpikes_triggered()
        elif key == Qt.Key_B:
            sw.on_actionAlignBest_triggered()

    @QtCore.pyqtSlot()
    def on_actionUndo_triggered(self):
        """Undo button click. Undo previous cluster change"""
        try:
            cc = self.cchanges[self.cci]
        except IndexError:
            print('Nothing to undo')
            return
        print('Undoing: %s' % cc.message)
        self.ApplyClusterChange(cc, direction='back')
        self.cci -= 1 # move pointer one change back on the stack
        print('Undo complete')

    @QtCore.pyqtSlot()
    def on_actionRedo_triggered(self):
        """Redo button click. Redo next cluster change"""
        try:
            cc = self.cchanges[self.cci+1]
        except IndexError:
            print('Nothing to redo')
            return
        print('Redoing: %s' % cc.message)
        self.ApplyClusterChange(cc, direction='forward')
        self.cci += 1 # move pointer one change forward on the stack
        print('Redo complete')

    @QtCore.pyqtSlot()
    def on_actionSpikeWindow_triggered(self):
        """Spike window toggle menu/button event"""
        self.ToggleWindow('Spike')

    @QtCore.pyqtSlot()
    def on_actionChartWindow_triggered(self):
        """Chart window toggle menu/button event"""
        self.ToggleWindow('Chart')

    @QtCore.pyqtSlot()
    def on_actionLFPWindow_triggered(self):
        """LFP window toggle menu/button event"""
        self.ToggleWindow('LFP')

    @QtCore.pyqtSlot()
    def on_actionSortWindow_triggered(self):
        """Sort window toggle menu/button event"""
        self.ToggleWindow('Sort')

    @QtCore.pyqtSlot()
    def on_actionClusterWindow_triggered(self):
        """Cluster window toggle menu/button event"""
        self.ToggleWindow('Cluster')

    @QtCore.pyqtSlot()
    def on_actionMPLWindow_triggered(self):
        """Matplotlib window toggle menu/button event"""
        self.ToggleWindow('MPL')

    @QtCore.pyqtSlot()
    def on_actionShell_triggered(self):
        """Shell window toggle menu/button event"""
        #self.ToggleWindow('Shell')
        # FIXME: this blocks until you Ctrl-D out of ipython:
        ipshell()

    @QtCore.pyqtSlot()
    def on_actionRasters_triggered(self):
        """Spike rasters toggle menu event"""
        self.ToggleRasters()

    @QtCore.pyqtSlot()
    def on_actionTimeRef_triggered(self):
        """Time reference toggle menu event"""
        self.ToggleRef('TimeRef')

    @QtCore.pyqtSlot()
    def on_actionVoltageRef_triggered(self):
        """Voltage reference toggle menu event"""
        self.ToggleRef('VoltageRef')

    @QtCore.pyqtSlot()
    def on_actionScale_triggered(self):
        """Scale toggle menu event"""
        self.ToggleRef('Scale')

    @QtCore.pyqtSlot()
    def on_actionCaret_triggered(self):
        """Caret toggle menu event"""
        self.ToggleRef('Caret')

    @QtCore.pyqtSlot()
    def on_actionFiltmethNone_triggered(self):
        """None filtering menu choice event"""
        self.SetFiltmeth(None)

    @QtCore.pyqtSlot()
    def on_actionFiltmethBW_triggered(self):
        """Butterworth filtering menu choice event"""
        self.SetFiltmeth('BW')

    @QtCore.pyqtSlot()
    def on_actionFiltmethBWNC_triggered(self):
        """Non-causal Butterworth filtering menu choice event"""
        self.SetFiltmeth('BWNC')

    @QtCore.pyqtSlot()
    def on_actionFiltmethWMLDR_triggered(self):
        """WMLDR filtering menu choice event"""
        self.SetFiltmeth('WMLDR')

    @QtCore.pyqtSlot()
    def on_actionCARNone_triggered(self):
        """None CAR menu choice event"""
        self.SetCAR(None)

    @QtCore.pyqtSlot()
    def on_actionCARMedian_triggered(self):
        """Median CAR menu choice event"""
        self.SetCAR('Median')

    @QtCore.pyqtSlot()
    def on_actionCARMean_triggered(self):
        """Mean CAR menu choice event"""
        self.SetCAR('Mean')

    @QtCore.pyqtSlot()
    def on_action20kHz_triggered(self):
        """20kHz menu choice event"""
        self.SetSampfreq(20000)

    @QtCore.pyqtSlot()
    def on_action25kHz_triggered(self):
        """25kHz menu choice event"""
        self.SetSampfreq(25000)

    @QtCore.pyqtSlot()
    def on_action30kHz_triggered(self):
        """30kHz menu choice event"""
        self.SetSampfreq(30000)

    @QtCore.pyqtSlot()
    def on_action40kHz_triggered(self):
        """40kHz menu choice event"""
        self.SetSampfreq(40000)

    @QtCore.pyqtSlot()
    def on_action50kHz_triggered(self):
        """50kHz menu choice event"""
        self.SetSampfreq(50000)

    @QtCore.pyqtSlot()
    def on_action60kHz_triggered(self):
        """60kHz menu choice event"""
        self.SetSampfreq(60000)

    @QtCore.pyqtSlot()
    def on_action80kHz_triggered(self):
        """80kHz menu choice event"""
        self.SetSampfreq(80000)

    @QtCore.pyqtSlot()
    def on_action100kHz_triggered(self):
        """100kHz menu choice event"""
        self.SetSampfreq(100000)

    @QtCore.pyqtSlot()
    def on_action120kHz_triggered(self):
        """120kHz menu choice event"""
        self.SetSampfreq(120000)

    @QtCore.pyqtSlot()
    def on_actionSampleAndHoldCorrect_triggered(self):
        """Sample & hold menu event"""
        enable = self.ui.actionSampleAndHoldCorrect.isChecked()
        self.SetSHCorrect(enable)

    #def onFilePosLineEdit_textChanged(self, text): # updates immediately
    def on_filePosLineEdit_editingFinished(self): # updates on Enter/loss of focus
        text = str(self.ui.filePosLineEdit.text())
        try:
            t = self.str2t[text]
        except KeyError: # convert to float to allow exp notation shorthand
            t = float(text)
        self.seek(t)

    @QtCore.pyqtSlot()
    def on_actionAboutSpyke_triggered(self):
        with open('../LICENSE', 'r') as lf:
            LICENSE = lf.read()
        system = """<p>Python %s, Qt %s, PyQt %s<br>
                    %s</p>""" % (platform.python_version(),
                                 QtCore.QT_VERSION_STR, QtCore.PYQT_VERSION_STR,
                                 platform.platform())
        text = """
        <h2><a href=http://spyke.github.io>spyke</a> %s</h2>
        <p>A tool for neuronal waveform visualization and spike sorting</p>

        <p>Copyright &copy; 2008-2017 <a href=http://mspacek.github.io>Martin Spacek</a>,
                                                                       Reza Lotun<br>
           <a href=http://swindale.ecc.ubc.ca>Swindale</a> Lab,
           University of British Columbia, Vancouver, Canada<br>
           
           <a href=http://www.neuro.bio.lmu.de/members/system_neuro_busse/busse_l/index.html>
           Busse</a> Lab, Ludwig-Maximilians-University, Munich, Germany</p>

        <p>Some functionality inherited from Tim Blanche's Delphi program "SurfBawd".</p>

        <p>Many icons were copied from Ubuntu's <a
        href=http://launchpad.net/humanity>Humanity</a> icon theme.</p>

        <p>%s</p>

        %s""" % (__version__, LICENSE, system)
        QtGui.QMessageBox.about(self, "About spyke", text)

    @QtCore.pyqtSlot()
    def on_actionAboutQt_triggered(self):
        QtGui.QMessageBox.aboutQt(self)

    @QtCore.pyqtSlot()
    def on_filePosStartButton_clicked(self):
        self.seek(self.str2t['start'])

    @QtCore.pyqtSlot()
    def on_filePosEndButton_clicked(self):
        self.seek(self.str2t['end'])

    @QtCore.pyqtSlot(int)
    def on_slider_valueChanged(self, slideri):
        t = slideri * self.hpstream.tres
        self.seek(t)

    def update_slider(self):
        """Update slider limits and step sizes. Slider ticks are multiples of tres"""
        tres = self.hpstream.tres
        self.ui.slider.setRange(intround(self.trange[0] / tres),
                                intround(self.trange[1] / tres))
        self.ui.slider.setValue(intround(self.t / tres))
        self.ui.slider.setSingleStep(1)
        self.ui.slider.setPageStep(intround((self.spiketw[1]-self.spiketw[0]) / tres))
        self.ui.slider.setInvertedControls(True)

    @QtCore.pyqtSlot()
    def on_detectButton_clicked(self):
        """Detect pane Detect button click"""
        sort = self.CreateNewSort() # create a new sort, with bound stream
        self.get_detector() # update Sort's current detector with new one from widgets
        if sort.detector.extractparamsondetect:
            self.init_extractor() # init the Extractor
        # create struct array of spikes and 3D array of spike waveform data:
        sort.spikes, sort.wavedata = sort.detector.detect(logpath=self.streampath)
        sort.update_usids()

        # lock down filtmeth, car, sampfreq and shcorrect attribs:
        sort.filtmeth = sort.stream.filtmeth
        sort.car = sort.stream.car
        sort.sampfreq = sort.stream.sampfreq
        sort.shcorrect = sort.stream.shcorrect

        self.ui.progressBar.setFormat("%d spikes" % sort.nspikes)
        self.EnableSortWidgets(True)
        sw = self.OpenWindow('Sort') # ensure it's open
        if sort.nspikes > 0:
            self.on_plotButton_clicked()

    def init_extractor(self):
        """Initialize Extractor"""
        #XYmethod = self.XY_extract_radio_box.GetStringSelection()
        # hard code XYmethod for now, don't really need extract pane:
        if self.sort.probe.ncols == 1:
            XYmethod = 'Gaussian 1D'
        else:
            XYmethod = 'Gaussian 2D'
        # create Extractor, or eventually, call a self.get_extractor() method instead:
        ext = Extractor(self.sort, XYmethod, maxsigma=self.sort.detector.inclr)
        self.sort.extractor = ext
        # eventually, update extractor from multiple Extract pane widgets:
        #self.update_extractor(ext)

    def OnXYExtract(self, evt=None):
        """Extract pane XY Extract button click. Extracts (or re-extracts and
        overwrites) XY parameters from all sort.spikes, and stores
        them as spike attribs"""
        try:
            self.sort.extractor
        except AttributeError:
            self.init_extractor()

        #import cProfile
        #cProfile.runctx('self.sort.extractor.extract_all_XY()', globals(), locals())

        self.sort.extractor.extract_all_XY() # adds extracted XY params to sort.spikes
        self.windows['Sort'].uslist.updateAll() # update any columns showing param values
        self.EnableSpikeWidgets(True) # enable cluster_pane

    def OnWaveletExtract(self, evt=None):
        """Extract pane wavelet Extract button click. Extracts (or re-extracts and
        overwrites) wavelet coefficients from all sort.spikes, and stores
        them as spike attribs"""
        try:
            self.sort.extractor
        except AttributeError:
            self.init_extractor()

        #import cProfile
        #cProfile.runctx('self.sort.extractor.extract_all_XY()', globals(), locals())

        # extract coeffs of selected wavelet type, add coeffs to sort.spikes
        wavelet = self.wavelet_extract_radio_box.GetStringSelection()
        self.sort.extractor.extract_all_wcs(wavelet)
        self.windows['Sort'].uslist.updateAll() # update any columns showing param values
        self.EnableSpikeWidgets(True) # enable cluster_pane

    def OnTemporalExtract(self, evt=None):
        """Extract pane temporal Extract button click. Extracts (or re-extracts and
        overwrites) temporal params from all sort.spikes, and stores
        them as spike attribs"""
        try:
            self.sort.extractor
        except AttributeError:
            self.init_extractor()

        self.sort.extractor.extract_all_temporal()
        self.windows['Sort'].uslist.updateAll() # update any columns showing param values
        self.EnableSpikeWidgets(True) # enable cluster_pane

    @QtCore.pyqtSlot()
    def on_clusterButton_clicked(self):
        """Cluster pane Cluster button click"""
        s = self.sort
        spikes = s.spikes
        #sids = self.GetAllSpikes() # all selected spikes
        # always cluster all spikes in existing clusters, don't just cluster some subset,
        # since existing clusters are always deleted in apply_clustering and
        # ApplyClusterChange, and spikes that aren't in that subset would inadvertantly
        # become unsorted
        sids = np.concatenate([self.GetClusterSpikes(), self.GetUnsortedSpikes()])
        sids.sort()
        oldclusters = self.GetClusters() # all selected clusters
        if len(sids) == 0: # nothing selected
            sids = spikes['id'] # all spikes (sorted)
            oldclusters = s.clusters.values() # all clusters
        dims = self.GetClusterPlotDims()
        comps = np.any([ dim.startswith('c') and dim[-1].isdigit() for dim in dims ])
        subsidss = [] # sids grouped into subclusters, each to be clustered separately
        msgs = []
        t0 = time.time()
        if comps and np.all(sids == spikes['id']): # doing PCA/ICA on all spikes
            if not oldclusters:
                print("No existing clusters to sequentially do PCA/ICA on and subcluster")
                return
            # partition data by existing clusters before clustering,
            # restrict to only clustered spikes:
            for oldcluster in oldclusters:
                subsidss.append(oldcluster.neuron.sids)
                msgs.append('oldcluster %d' % oldcluster.id)
            sids = np.concatenate(subsidss) # update
            sids.sort()
        else: # just the selected spikes
            subsidss.append(sids)
            msgs.append('%d selected sids' % len(sids))
        nids = self.subcluster(sids, subsidss, msgs, dims)
        print('Clustering took %.3f sec' % (time.time()-t0))
        self.apply_clustering(oldclusters, sids, nids, verb='GAC')

    def subcluster(self, sids, subsidss, msgs, dims):
        """Perform (sub)clustering according to subsids in subsidss. Incorporate results
        from each (sub)clustering into a single nids output array"""
        # init nids output array to be all unclustered:
        nids = np.zeros(len(sids), dtype=np.int32)
        for subsids, msg in zip(subsidss, msgs):
            print('Clustering %s on dims %r' % (msg, dims))
            subnids = self.gac(subsids, dims) # subclustering result
            ci = subnids > 0 # consider only the clustered sids
            subsids = subsids[ci]
            subnids = subnids[ci]
            nidoffset = max(nids) + 1
            nidsi = sids.searchsorted(subsids)
            nids[nidsi] = subnids + nidoffset
        return nids

    def chancombosplit(self):
        """Split spikes into clusters of unique channel combinations"""
        s = self.sort
        spikes = s.spikes
        sids = self.GetAllSpikes() # all selected spikes
        oldclusters = self.GetClusters() # all selected clusters
        if len(sids) == 0: # nothing selected
            sids = spikes['id'] # all spikes (sorted)
            oldclusters = s.clusters.values() # all clusters
        t0 = time.time()
        chans = spikes[sids]['chans']
        chans = tocontig(chans) # string view won't work without contiguity
        # each row becomes a string:
        strchans = chans.view('S%d' % (chans.itemsize*chans.shape[1]))
        # each row in uchancombos is a unique combination of chans:
        uchancombos = np.unique(strchans).view(chans.dtype).reshape(-1, chans.shape[1])
        if len(uchancombos) == 1:
            print("Selected spikes all share the same set of channels, can't chancombosplit")
            return
        # init to unclustered, shouldn't be any once done:
        nids = np.zeros(len(sids), dtype=np.int32)
        for comboi, chancombo in enumerate(uchancombos):
            nids[(chans == chancombo).all(axis=1)] = comboi + 1
        if (nids == 0).any():
            raise RuntimeError("there shouldn't be any unclustered points from chancombosplit")
        print('chancombosplit took %.3f sec' % (time.time()-t0))
        self.apply_clustering(oldclusters, sids, nids, verb='chancombo split')

    def maxchansplit(self):
        """Split spikes into clusters by maxchan"""
        s = self.sort
        spikes = s.spikes
        sids = self.GetAllSpikes() # all selected spikes
        oldclusters = self.GetClusters() # all selected clusters
        if len(sids) == 0: # nothing selected
            sids = spikes['id'] # all spikes (sorted)
            oldclusters = s.clusters.values() # all clusters
        t0 = time.time()
        maxchans = spikes[sids]['chan']
        umaxchans = np.unique(maxchans)
        if len(umaxchans) == 1:
            print("Selected spikes all share the same set of max channels, can't maxchansplit")
            return
        # init to unclustered, shouldn't be any once done:
        nids = np.zeros(len(sids), dtype=np.int32)
        for maxchani, maxchan in enumerate(umaxchans):
            nids[maxchans == maxchan] = maxchani + 1
        if (nids == 0).any():
            raise RuntimeError("there shouldn't be any unclustered points from maxchansplit")
        print('maxchansplit took %.3f sec' % (time.time()-t0))
        self.apply_clustering(oldclusters, sids, nids, verb='maxchan split')

    def densitysplit(self):
        """Split cluster pair by density along line between their centers in current
        cluster space"""
        s = self.sort
        spikes = s.spikes
        oldclusters = self.GetClusters() # all selected clusters
        if len(oldclusters) != 2:
            print("Need to select exactly 2 clusters to split them by density")
            return
        dims = self.GetClusterPlotDims()
        try:
            X, sids = self.get_param_matrix(dims=dims)
        except RuntimeError as err:
            print(err)
            return

        nids = s.spikes['nid'][sids] # copy
        unids = np.unique(nids)
        assert len(unids) == 2
        # centers of both clusters, use median:
        i0 = nids == unids[0]
        i1 = nids == unids[1]
        c0 = np.median(X[i0], axis=0) # ndims vector
        c1 = np.median(X[i1], axis=0)
        # line connecting the centers of the two clusters, wrt c0
        line = c1-c0
        line /= np.linalg.norm(line) # make it unit length
        #print('c0=%r, c1=%r, line=%r' % (c0, c1, line))
        proj = np.dot(X-c0, line) # projection of each point onto line
        nbins = max(intround(np.sqrt(len(proj))), 2) # good heuristic
        #print('nbins = %d' % nbins)
        hist, edges = np.histogram(proj, bins=nbins)
        ei0, ei1 = edges.searchsorted((np.median(proj[i0]), np.median(proj[i1])))
        # find histogram min between cluster medians:
        threshi = hist[ei0:ei1].argmin()
        thresh = edges[ei0:ei1][threshi]
        #print('thresh is %.3f' % thresh)
        #print('ei0, ei1: %d, %d' % (ei0, ei1))
        assert ei0 < ei1 # think this is always the case because projections are wrt c0
        nids[proj < thresh] = unids[0] # overwrite nid values in nids, since it's a copy
        nids[proj >= thresh] = unids[1]
        self.apply_clustering(oldclusters, sids, nids, verb='density split')

    def randomsplit(self):
        """Randomly split each selected cluster in half. This is done to increase
        gac() speed"""
        oldclusters = self.GetClusters() # all selected clusters
        subsidss = []
        for cluster in oldclusters:
            subsidss.append(cluster.neuron.sids)
        sids = np.concatenate(subsidss)
        sids.sort()
        destsubsidss = []
        for subsids in subsidss:
            np.random.shuffle(subsids) # shuffle in-place
            spliti = len(subsids) // 2
            destsubsids0 = subsids[:spliti]
            destsubsids0.sort() # sids should always go out sorted
            destsubsidss.append(destsubsids0)
            destsubsids1 = subsids[spliti:]
            destsubsids1.sort()
            destsubsidss.append(destsubsids1)
        # init to unclustered, shouldn't be any once done:
        nids = np.zeros(len(sids), dtype=np.int32)
        for i, destsubsids in enumerate(destsubsidss):
            nids[sids.searchsorted(destsubsids)] = i + 1
        if (nids == 0).any():
            raise RuntimeError("there shouldn't be any unclustered points from randomsplit")
        self.apply_clustering(oldclusters, sids, nids, verb='randomly split')

    def gac(self, sids, dims):
        """Cluster sids along dims, using NVS's gradient ascent algorithm"""
        s = self.sort
        norm = self.ui.normButton.isChecked()
        data, sids = self.get_param_matrix(sids=sids, dims=dims, norm=norm, scale=True)
        data = tocontig(data) # ensure it's contiguous for gac()
        # grab gac() params and run it
        self.update_sort_from_cluster_pane()
        npoints, ndims = data.shape
        print('Clustering %d points in %d-D space' % (npoints, ndims))
        t0 = time.time()
        nids = gac(data, sigma=s.sigma, rmergex=s.rmergex, rneighx=s.rneighx,
                   alpha=s.alpha, maxgrad=s.maxgrad,
                   maxnnomerges=1000, minpoints=s.minpoints)
        # nids from gac() are 0-based, but we want our single unit nids to be 1-based,
        # to leave room for junk cluster at 0 and multiunit clusters at nids < 0. So add 1:
        nids += 1
        print('GAC took %.3f sec' % (time.time()-t0))
        return nids
    
    def get_selchans(self, sids):
        """Return user selected chans. If none, automatically select and
        return chans within a radius encompassing 95% percent of sx values in sids,
        centered on average position of sids. Could also use a multiple of this
        derived sx to select more or fewer chans"""
        spikes = self.sort.spikes
        panel = self.windows['Sort'].panel
        selchans = panel.chans_selected # a list
        selchans.sort()
        if selchans and panel.manual_selection:
            return selchans # always return whatever's manually selected
        sxs = spikes['sx'][sids]
        sxs = np.sort(sxs) # get a sorted copy
        sxi = int(len(sxs) * 0.95) # round down, index > ~95% percent of values
        sx = sxs[sxi]
        dm = self.sort.detector.dm # DistanceMatrix
        spos = np.vstack((spikes['x0'][sids], spikes['y0'][sids])).T # sids x 2
        meanpos = spos.mean(axis=0) # mean spike position
        chanpos = np.asarray(dm.coords) # positions of enabled chans
        # Euclidean chan distances from meanpos:
        d = np.sqrt(np.sum((chanpos - meanpos)**2, axis=1))
        selchans = sorted(dm.chans[d <= sx]) # chans within sx of meanpos
        print('selection center: %.1f, %.1f um' % (meanpos[0], meanpos[1]))
        print('selection radius: %.1f um' % sx)
        panel.chans_selected = selchans
        panel.update_selvrefs()
        panel.draw_refs()
        panel.manual_selection = False
        return selchans

    def apply_clustering(self, oldclusters, sids, nids, verb=''):
        """Delete old clusters and replace the existing clustering of the desired sids
        with their new nids"""
        s = self.sort
        spikes = s.spikes
        sw = self.windows['Sort']
        cw = self.windows['Cluster']
        
        # deselect all clusters before potentially deleting any unselected
        # clusters, to avoid lack of Qt selection event when selection values
        # (not rows) change. Also, deselect usids while we're at it:
        self.SelectClusters(s.clusters, on=False)
        sw.uslist.clearSelection()

        # delete junk cluster if it exists and isn't in oldclusters,
        # add this deletion to cluster change stack
        if 0 in s.clusters and 0 not in [ c.id for c in oldclusters ]:
            # save some undo/redo stuff
            message = 'delete junk cluster 0'
            cc = ClusterChange(s.neurons[0].sids, spikes, message)
            cc.save_old([s.clusters[0]], s.norder, s.good)
            # delete it
            s.remove_neuron(0)
            # save more undo/redo stuff
            cc.save_new([], s.norder, s.good)
            self.AddClusterChangeToStack(cc)
            print(cc.message)

        # save some undo/redo stuff
        message = '%s clusters %r' % (verb, [ c.id for c in oldclusters ])
        cc = ClusterChange(sids, spikes, message)
        cc.save_old(oldclusters, s.norder, s.good)

        # start insertion indices of new clusters from first selected cluster, if any
        unids = np.unique(nids)
        nnids = len(unids)
        insertis = [None] * nnids
        if len(oldclusters) > 0:
            startinserti = s.norder.index(oldclusters[0].id)
            insertis = range(startinserti, startinserti+nnids)

        # delete old clusters
        self.DelClusters(oldclusters, update=False)

        # apply new clusters
        newclusters = []
        for nid, inserti in zip(unids, insertis):
            ii, = np.where(nids == nid)
            nsids = sids[ii] # sids belonging to this nid
            if nid != 0:
                nid = None # auto generate a new nid
            cluster = self.CreateCluster(update=False, id=nid, inserti=inserti)
            newclusters.append(cluster)
            neuron = cluster.neuron
            sw.MoveSpikes2Neuron(nsids, neuron, update=False)
            if len(nsids) == 0:
                raise RuntimeError('WARNING: neuron %d has no spikes for some reason'
                                   % neuron.id)
            cluster.update_pos()

        # save more undo/redo stuff
        cc.save_new(newclusters, s.norder, s.good)
        self.AddClusterChangeToStack(cc)

        # now do some final updates
        self.UpdateClustersGUI()
        if not np.all(sids == spikes['id']): # if clustering only some spikes,
            self.SelectClusters(newclusters) # select all newly created cluster(s)
        if np.all(sids == cw.glWidget.sids):
            self.ColourPoints(newclusters) # just recolour
        else:
            self.on_plotButton_clicked() # need to do a full replot
        cc.message += ' into %r' % [c.id for c in newclusters]
        print(cc.message)

    @QtCore.pyqtSlot()
    def on_x0y0VppButton_clicked(self):
        """Cluster pane x0y0Vpp button click. Set plot dims to x0, y0, and Vpp"""
        self.SetPlotDims('x0', 'y0', 'Vpp')

    @QtCore.pyqtSlot()
    def on_c0c1c2Button_clicked(self):
        """Cluster pane c0c1c2 button click. Set plot dims to c0, c1, and c2"""
        s = self.sort
        ctrl = QtGui.QApplication.instance().keyboardModifiers() == Qt.ControlModifier
        if ctrl:
            try:
                del s.X[s.get_Xhash(*self.get_Xhash_args())] # force recalc
            except (AttributeError, KeyError): pass
        self.SetPlotDims('c0', 'c1', 'c2')

    @QtCore.pyqtSlot()
    def on_c0c1tButton_clicked(self):
        """Cluster pane c0c1t button click. Set plot dims to c0, c1, and t"""
        s = self.sort
        ctrl = QtGui.QApplication.instance().keyboardModifiers() == Qt.ControlModifier
        if ctrl:
            try:
                del s.X[s.get_Xhash(*self.get_Xhash_args())] # force recalc
            except (AttributeError, KeyError): pass
        self.SetPlotDims('c0', 'c1', 't')

    def SetPlotDims(self, x, y, z):
        """Set plot dimensions to x, y, z, and replot"""
        xi = self.ui.xDimComboBox.findText(x)
        yi = self.ui.yDimComboBox.findText(y)
        zi = self.ui.zDimComboBox.findText(z)        
        self.ui.xDimComboBox.setCurrentIndex(xi)
        self.ui.yDimComboBox.setCurrentIndex(yi)
        self.ui.zDimComboBox.setCurrentIndex(zi)
        self.on_plotButton_clicked() # replot

    def get_param_matrix(self, sids=None, dims=None, norm=False, scale=True):
        """Given list of dims, get clustering parameter matrix according to
        current selection of sids and channels"""
        s = self.sort
        sw = self.OpenWindow('Sort') # in case it isn't already open
        cw = self.OpenWindow('Cluster') # in case it isn't already open
        comps = np.any([ dim.startswith('c') and dim[-1].isdigit() for dim in dims ])
        # calc RMS error between each spike and its clusters median waveform, if any?
        rmserror = np.any([ dim == 'RMSerror' for dim in dims ])
        if sids is None:
            sids = self.GetAllSpikes() # only selected spikes
        if len(sids) == 0: # if none selected
            if comps: # if component analysis selected
                raise RuntimeError('need non-empty spike selection to do component analysis')
            else: # return all spike ids
                sids = self.sort.spikes['id']
        kind = None
        tis = None
        selchans = None
        if comps or rmserror:
            tis = sw.tis # waveform time indices to include, centered on spike
            selchans = self.get_selchans(sids)
        if comps:
            kind = str(self.ui.componentAnalysisComboBox.currentText())
        norm = self.ui.normButton.isChecked()
        X = s.get_param_matrix(kind=kind, sids=sids, tis=tis, selchans=selchans,
                               norm=norm, dims=dims, scale=scale)
        return X, sids

    def get_Xhash_args(self):
        """Return currently selected clustering paramaters that would be used to generate the
        identifying hash for the dimension reduced matrix if it were to be calculated at this
        point in time"""
        sw = self.OpenWindow('Sort') # in case it isn't already open
        kind = str(self.ui.componentAnalysisComboBox.currentText())
        sids = self.GetAllSpikes() # only selected spikes
        tis = sw.tis # waveform time indices to include, centered on spike
        selchans = np.asarray(self.get_selchans(sids))
        chans = self.sort.get_common_chans(sids, selchans)[0]
        npcsperchan = self.sort.npcsperchan
        norm = self.ui.normButton.isChecked()
        return kind, sids, tis, chans, npcsperchan, norm

    @QtCore.pyqtSlot()
    def on_plotButton_clicked(self):
        """Cluster pane plot button click. Plot points and colour them
        according to their clusters."""
        s = self.sort
        ctrl = QtGui.QApplication.instance().keyboardModifiers() == Qt.ControlModifier
        if ctrl:
            try:
                del s.X[s.get_Xhash(*self.get_Xhash_args())] # force recalc
            except (AttributeError, KeyError): pass
        cw = self.OpenWindow('Cluster') # in case it isn't already open
        dims = self.GetClusterPlotDims()
        try:
            X, sids = self.get_param_matrix(dims=dims)
        except RuntimeError as err:
            print(err)
            return
        if len(X) == 0:
            return # nothing to plot
        nids = s.spikes['nid'][sids]
        cw.plot(X, sids, nids)
        sw = self.OpenWindow('Sort') # in case it isn't already open
        sw.PlotClusterHistogram(X, nids) # auto update cluster histogram plot

    @QtCore.pyqtSlot()
    def on_normButton_clicked(self):
        """Cluster pane norm button click"""
        if self.ui.normButton.isChecked():
            print('Normalizing spike amplitudes')
        else:
            print('Un-normalizing spike amplitudes')
        self.windows['Sort'].panel.updateAllItems() # refresh plotted waveforms
        self.on_plotButton_clicked() # refresh cluster plot

    @QtCore.pyqtSlot()
    def get_cleaning_density_hist(self):
        """Calculate histogram of point densities of selected spikes over selected
        clustering dimensions from origin"""
        dims = self.GetClusterPlotDims()
        X, sids = self.get_param_matrix(dims=dims)
        # each dim in X has 0 mean, so X is centered on origin
        X = np.float64(X) # convert to double precision
        ndims = X.shape[1]
        r = np.sqrt(np.square(X).sum(axis=1)) # all +ve values
        r /= r.std() # normalize to unit variance
        nbins = intround(np.sqrt(len(X))) # good heuristic
        rhist, edges = np.histogram(r, nbins) # distance hist, edges includes the right edge
        ledges = edges[:-1] # keep just the left edges, discard the last right edge
        assert len(ledges) == nbins
        binwidth = ledges[1] - ledges[0]
        # density histogram: npoints / fractional volume        
        dhist = np.float64(rhist) / np.diff(edges**ndims)
        dhist /= (dhist * binwidth).sum() # normalize to unit area
        return dhist, ledges, binwidth, ndims, sids, r

    @QtCore.pyqtSlot()
    def on_cleanHistButton_clicked(self):
        """Cluster pane cleaning hist button click. Plot histogram of point
        densities of selected spikes over selected clustering dimensions from origin,
        compare to Gaussian. Note that each time you reject points > nstds away
        from origin, the distrib may get less and less Gaussian, and more and more
        uniform"""
        dhist, ledges, binwidth, ndims, sids, r = self.get_cleaning_density_hist()
        ris = ledges + (binwidth / 2) # center values of bins
        gauss = g(0, 1, ris)
        gauss /= (gauss * binwidth).sum() # normalize to unit area
        djs = DJS(dhist, gauss)
        mplw = self.OpenWindow('MPL')
        a = mplw.ax
        a.clear()
        mplw.setWindowTitle('Density Histogram')
        a.bar(ledges, dhist, width=binwidth)
        a.plot(ris, gauss, '-') # plot Gaussian on top of density histogram
        a.set_title('%dD cluster density histogram, DJS = %.3f' % (ndims, djs))
        a.set_xlabel('nstdevs')
        a.set_ylabel('normalized density')
        mplw.f.tight_layout(pad=0.3) # crop figure to contents
        mplw.figurecanvas.draw()

    @QtCore.pyqtSlot()
    def on_cleanButton_clicked(self):
        """Cluster pane clean button click. Set as unsorted those points that fall outside
        of nstds distance away in the cluster density histogram plotted above"""
        # r vals are in nstds units:
        dhist, ledges, binwidth, ndims, sids, r = self.get_cleaning_density_hist()
        nstds = self.ui.cleanNstdsSpinBox.value()
        nids = self.sort.spikes[sids]['nid']
        unids = np.unique(nids)
        oldclusters = [ self.sort.clusters[unid] for unid in unids ]
        nids[r > nstds] = 0 # set some sids to cluster 0, ie unclustered
        self.apply_clustering(oldclusters, sids, nids, verb='clean')

    @QtCore.pyqtSlot()
    def on_calcMatchErrorsButton_clicked(self):
        """Match pane calc button click. Calculate rmserror between all clusters and
        all unsorted spikes. Also calculate which cluster each unsorted spike matches best"""
        spikes = self.sort.spikes
        wavedata = self.sort.wavedata
        cids = np.sort(self.sort.clusters.keys())
        sids = self.sort.usids.copy()
        ncids, nsids = len(cids), len(sids)
        print('Calculating rmserror between all %d clusters and all %d unsorted spikes'
              % (ncids, nsids))
        errs = np.empty((ncids, nsids), dtype=np.float32)
        errs.fill(np.inf) # TODO: replace with sparse matrix with np.inf as default value
        for cidi, cid in enumerate(cids):
            neuron = self.sort.neurons[cid]
            for sidi, sid in enumerate(sids):
                chan = spikes['chan'][sid]
                nchans = spikes['nchans'][sid]
                chans = spikes['chans'][sid][:nchans]
                # TODO: this is a bit wasteful if no chans are in common:
                sdata = wavedata[sid, :nchans]
                try:
                    ndata, sdata = neuron.getCommonWaveData(chan, chans, sdata)
                except ValueError: # not comparable
                    continue
                errs[cidi, sidi] = core.rms(ndata - sdata)
        errs = self.sort.converter.AD2uV(errs) # convert from AD units to uV, np.infs are OK
        self.match = Match(cids, sids, errs)
        print('Done calculating rmserror between all %d clusters and all %d unsorted spikes'
              % (ncids, nsids))
        return self.match
        
    @QtCore.pyqtSlot()
    def on_plotMatchErrorsButton_clicked(self):
        """Match pane plot match errors button click. Plot histogram of rms error between
        current cluster and all unclustered spikes that best fit the current cluster"""
        cluster = self.GetCluster()
        cid = cluster.id
        if not hasattr(self, 'match') or self.match == None:
            self.match = self.on_calcMatchErrorsButton_clicked() # (re)calc
        errs = self.match.get_best_errs(cid)
        if len(errs) == 0:
            print('No unsorted spikes fit cluster %d' % cid)
            return
        f = pl.gcf()
        pl.clf()
        f.canvas.parent().setWindowTitle('cluster %d rmserror histogram' % cid)
        binsize = self.ui.matchErrorPlotBinSizeSpinBox.value()
        pl.hist(errs, bins=np.arange(0, 50, binsize))
        pl.title('rmserrors between cluster %d and %d unsorted spikes' %
                 (cid, len(errs)))
        pl.xlabel('rmserror (uV)')
        pl.ylabel('count')
        
    @QtCore.pyqtSlot()
    def on_matchButton_clicked(self):
        """Deselect any selected unsorted spikes in uslist, and then select
        unsorted spikes that fall below match error threshold and fit the
        current cluster best"""
        cluster = self.GetCluster()
        cid = cluster.id
        if not hasattr(self, 'match') or self.match == None:
            self.match = self.on_calcMatchErrorsButton_clicked() # (re)calc
        errs = self.match.get_best_errs(cid)
        if len(errs) == 0:
            print('No unsorted spikes fit cluster %d' % cid)
            return
        bestsids = self.match.best[cid]
        thresh = self.ui.matchThreshSpinBox.value()
        sids = bestsids[errs <= thresh]
        sidis = self.sort.usids.searchsorted(sids)
        # clear uslist selection, select sidis rows in uslist
        sw = self.windows['Sort']
        sw.uslist.clearSelection()
        sw.uslist.selectRows(sidis, on=True, scrollTo=False)
        print('Matched %d spikes to cluster %d' % (len(sids), cid))

    @QtCore.pyqtSlot()
    def on_plotXcorrsButton_clicked(self):
        """Plot all cross/auto correlograms for all selected neurons, and display
        them in an upper or lower triangle configuration"""
        ## TODO: for now, just plot a single cross/auto correlogram
        clusters = self.GetClusters()
        xsids = clusters[0].neuron.sids
        if len(clusters) == 1:
            autocorr = True
            ysids = xsids # x and y are identical
        elif len(clusters) == 2:
            autocorr = False
            ysids = clusters[1].neuron.sids
        else:
            raise NotImplementedError("can't deal with more than one xcorr for now")
        xspikets = self.sort.spikes['t'][xsids]
        yspikets = self.sort.spikes['t'][ysids]

        ## TODO: spikes['t'][sids] is very different from spikes[sids]['t'] !
        ## The first is C contig, the second is not! The first probably makes a copy,
        ## while the second does not. First is much much faster for array ops, while
        ## the second conserves memory, and avoids needless copying, which might be faster
        ## if no array ops are involved. Should check all the code that pulls stuff out of
        ## the spikes recarray, and choose the best one more carefully!
        
        trange = self.ui.xcorrsRangeSpinBox.value() * 1000 # convert to us
        trange = max(1000, trange) # enforce min trange, in us
        trange = np.array([-trange, trange]) # convert to a +/- array, in us
        
        t0 = time.time()
        dts = util.xcorr(xspikets, yspikets, trange=trange) # in us
        print('xcorr calc took %.3f sec' % (time.time()-t0))
        if autocorr:
            dts = dts[dts != 0] # remove 0s for autocorr
        #print(dts)

        dts = dts / 1000 # in ms, converts to float64 array
        trange = trange / 1000 # in ms, converts to float64 array
        nbins = intround(np.sqrt(len(dts))) # good heuristic
        nbins = max(20, nbins) # enforce min nbins
        nbins = min(100, nbins) # enforce max nbins
        t = np.linspace(start=trange[0], stop=trange[1], num=nbins, endpoint=True)
        n = np.histogram(dts, bins=t, density=False)[0]
        binwidth = t[1] - t[0] # all should be equal width

        # plot:
        mplw = self.OpenWindow('MPL')
        a = mplw.ax
        a.clear()
        # omit last right edge in t:
        a.bar(t[:-1], height=n, width=binwidth, color='k', edgecolor='k')
        a.set_xlim(t[0], t[-1])
        a.set_xlabel('ISI (ms)')
        a.set_ylabel('count')
        if autocorr:
            windowtitle = "n%d autocorr" % clusters[0].id
        else:
            windowtitle = "n%d xcorr with n%d" % (clusters[0].id, clusters[1].id)
        mplw.setWindowTitle(windowtitle)
        title = windowtitle + ', binwidth: %.2f ms' % binwidth
        print(title)
        a.set_title(title)
        #a.set_ylabel('ISI rate (Hz)')
        mplw.f.tight_layout(pad=0.3) # crop figure to contents
        mplw.figurecanvas.draw()

    @QtCore.pyqtSlot()
    def on_ISICleanButton_clicked(self):
        """If only one cluster is selected, split off any duplicate spikes within that
        cluster, according to the ISI threshold. If multiple clusters or no clusters are
        selected, remove any duplicate spikes within selected clusters or all clusters,
        respectively, according to the same single ISI threshold. As implemented, the latter
        is not undoable"""
        clusters = self.GetClusters()
        minISI = self.ui.minISISpinBox.value()
        spikes = self.sort.spikes
        nids = [ cluster.id for cluster in clusters ] # selected neurons, in norder
        if len(nids) == 0: # if no neurons selected, clean all neurons
            nids = sorted(self.sort.neurons)

        rmsidss = {} # dict of lists of sids to split off or remove, indexed by nid
        print('Duplicate spikes:')
        for nid in nids:
            # For each pair of duplicate spikes, keep whichever has the most channel overlap
            # with neuron template. If they have same amount of overlap, keep the first one
            neuron = self.sort.neurons[nid]
            rmsids = [] # list of sids to remove for this neuron
            # pick out the first sid of each pair of duplicate sids, if any:
            sidis = np.where(np.diff(spikes['t'][neuron.sids]) <= minISI)[0]
            if len(sidis) == 0:
                continue # skip to next nid
            #x0, y0 = neuron.cluster.pos['x0'], neuron.cluster.pos['y0']
            for sidi in sidis:
                sid0 = neuron.sids[sidi] # 1st spike in each pair
                sid1 = neuron.sids[sidi+1] # 2nd spike in each pair
                nchans0 = spikes['nchans'][sid0]
                nchans1 = spikes['nchans'][sid1]
                chans0 = spikes['chans'][sid0][:nchans0]
                chans1 = spikes['chans'][sid1][:nchans1]
                ncommon0 = len(np.intersect1d(neuron.chans, chans0))
                ncommon1 = len(np.intersect1d(neuron.chans, chans1))
                if ncommon0 >= ncommon1:
                    # sid0 has more template chan overlap, or both are equal, keep sid0
                    rmsid = sid1
                else:
                    # sid1 has more template chan overlap, keep it
                    rmsid = sid0
                """
                # code for choosing the one closest to template mean position, not as robust:
                d02 = (spikes['x0'][sid] - x0)**2 + (spikes['y0'][sid] - y0)**2
                d12 = (spikes['x0'][sid+1] - x0)**2 + (spikes['y0'][sid+1] - y0)**2
                if d02 <= d12:
                    rmsid = sid + 1
                else:
                    rmsid = sid
                """
                rmsids.append(rmsid)
            print('neuron %d: %r' % (nid, rmsids))
            rmsidss[nid] = rmsids
        nrm = sum([ len(rmsids) for rmsids in rmsidss.values() ])
        print('Found %d duplicate spikes' % nrm)
        if nrm == 0:
            return
        sw = self.windows['Sort']
        if len(nids) == 1: # split duplicate spikes from single cluster into cluster 0
            sidis = neuron.sids.searchsorted(rmsids)
            sw.nslist.selectRows(sidis) # select spikes to split off from single cluster
            self.SplitSpikes(delete=True) # split them off into cluster 0 (undoable)
            return
        # otherwise, remove duplicate spikes from multiple clusters:
        val = QtGui.QMessageBox.question(self, "Remove %d duplicate spikes" % nrm,
              "Are you sure? This will clear the undo/redo stack, and is not undoable.",
              QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if val == QtGui.QMessageBox.No:
            return
        # do the actual removal:
        for nid, rmsids in rmsidss.items():
            neuron = self.sort.neurons[nid]
            neuron.sids = np.setdiff1d(neuron.sids, rmsids) # remove from source neuron
            spikes['nid'][rmsids] = 0 # set to junk in spikes struct array
            neuron.wave.data = None # trigger template mean update
            if neuron in sw.nslist.neurons:
                sw.nslist.neurons = sw.nslist.neurons # trigger nslist refresh
        # update usids and uslist:
        self.sort.update_usids()
        sw.uslist.updateAll()
        # cluster changes in stack no longer applicable, reset cchanges:
        del self.cchanges[:]
        print('Removed %d duplicate spikes' % nrm)

    def GetSortedSpikes(self):
        """Return IDs of selected sorted spikes"""
        sw = self.windows['Sort']
        srows = sw.nslist.selectedRows()
        return sw.nslist.sids[srows]

    def GetUnsortedSpikes(self):
        """Return IDs of selected unsorted spikes"""
        sw = self.windows['Sort']
        srows = sw.uslist.selectedRows()
        return self.sort.usids[srows]

    def GetClusterSpikes(self):
        """Return sorted IDs of all spikes of selected clusters"""
        clusters = self.GetClusters()
        if len(clusters) == 0:
            return np.array([], dtype=np.int64)
        sids = []
        for cluster in clusters:
            sids.append(cluster.neuron.sids)
        sids = np.concatenate(sids)
        sids.sort()
        return sids

    def GetSpikes(self):
        """Return IDs of explicitly selected spikes"""
        sw = self.windows['Sort']
        return np.concatenate([ self.GetSortedSpikes(), self.GetUnsortedSpikes() ])

    def GetSpike(self):
        """Return ID of just one selected spike, from nslist or uslist"""
        sids = self.GetSpikes()
        nselected = len(sids)
        if nselected != 1:
            raise RuntimeError("can't figure out which of the %d selected spike IDs you want"
                               % nselected)
        return sids[0]

    def GetAllSpikes(self):
        """Return sorted IDs of all selected spikes, whether explicitly or implicitly
        selected"""
        sids = []
        ssids = self.GetSortedSpikes()
        sids.append(ssids)
        # if no sorted spikes explicitly selected, check if any clusters are:
        if len(ssids) == 0:
            sids.append(self.GetClusterSpikes())
        # include any selected usids as well
        sids.append(self.GetUnsortedSpikes())
        sids = np.concatenate(sids)
        sids.sort()
        return sids

    def GetClusterIDs(self):
        """Return list of IDs of currently selected clusters, in norder"""
        sw = self.windows['Sort']
        cids = [ i.data().toInt()[0] for i in sw.nlist.selectedIndexes() ]
        #cids.sort() # don't do regular sort, sort by norder
        ciis = np.argsort([ self.sort.norder.index(cid) for cid in cids ])
        return [ cids[cii] for cii in ciis ] # in norder

    def GetClusters(self):
        """Return list of currently selected clusters, in norder"""
        cids = self.GetClusterIDs() # already in norder
        return [ self.sort.clusters[cid] for cid in cids ]

    def GetCluster(self):
        """Return just one selected cluster"""
        clusters = self.GetClusters()
        nselected = len(clusters)
        if nselected != 1:
            raise RuntimeError("can't figure out which of the %d selected clusters you want"
                               % nselected)
        return clusters[0]

    def SelectClusters(self, clusters, on=True):
        """Select/deselect clusters"""
        clusters = toiter(clusters)
        try:
            selnids = [ cluster.id for cluster in clusters ]
        except AttributeError: # assume they're ints
            selnids = [ cluster for cluster in clusters ]
        rows = [ self.sort.norder.index(selnid) for selnid in selnids ]
        nlist = self.windows['Sort'].nlist
        nlist.selectRows(rows, on)
        #print('Set rows %r to %r' % (rows, on))

    def ToggleCluster(self, cluster):
        """Toggle selection of given cluster"""
        sw = self.windows['Sort']
        try:
            nid = cluster.id
        except AttributeError: # assume it's an int
            nid = cluster
        row = self.sort.norder.index(nid)
        on = not sw.nlist.rowSelected(row)
        sw.nlist.selectRows(row, on=on)
        return on

    def SelectSpikes(self, sids, on=True):
        """Set selection state of given spikes, as well as their current clusters, if any"""
        sw = self.windows['Sort']
        nids = self.sort.spikes['nid'][sids]

        # select/deselect any unclustered spikes:
        usids = sids[nids == 0]
        if len(usids) > 0:
            usrows = self.sort.usids.searchsorted(usids)
            sw.uslist.selectRows(usrows, on=on)

        # select/deselect any clustered spikes, as well as their clusters:
        csids = sids[nids != 0] # clustered spike ids
        unids = np.unique(nids)
        unids = unids[unids != 0] # remove cluster 0
        # get currently selected sids in nslist, and the unids they belong to:
        selsids = sw.nslist.sids[sw.nslist.selectedRows()] # hopefully don't need a copy
        selunids = sw.nslist.nids
        if on == True: # find clustered spikes to add to selection:
            # add csids to selsids (get values in csids that aren't in selsids):
            csids = np.setdiff1d(csids, selsids, assume_unique=True) # to add
            allcsids = np.union1d(csids, selsids) # final
        elif on == False: # find clustered spikes to remove from selection:
            # remove csids from selsids:
            csids = np.intersect1d(csids, selsids, assume_unique=True) # to remove
            allcsids = np.setdiff1d(csids, selsids, assume_unique=True) # final
        else:
            raise ValueError("invalid 'on' value: %r" % on)
        if len(csids) == 0:
            return # no clustered spikes to add or remove
        newunids = np.unique(self.sort.spikes['nid'][allcsids]) # excludes cluster 0
        # select any new clusters so nslist has correct contents, this
        # changes contents of nslist and hence clears any currently selected sids:
        addunids = np.setdiff1d(newunids, selunids)
        if len(addunids) > 0:
            # all nids will be in sort.norder list, find their positions
            addnlistrows = [ self.sort.norder.index(unid) for unid in addunids ]
            sw.nlist.selectRows(addnlistrows, on=True)
        # now do the clustered spike selection:
        nslistrows = sw.nslist.sids.searchsorted(csids) # nslist.sids is sorted
        #t0 = time.time()
        sw.nslist.selectRows(nslistrows, on=on)
        #print('nslist.selectRows took %.3f sec' % (time.time()-t0))

    def CreateCluster(self, update=True, id=None, inserti=None):
        """Create a new cluster, add it to the GUI, return it"""
        s = self.sort
        neuron = s.create_neuron(id, inserti=inserti)
        sw = self.windows['Sort']
        if update:
            sw.nlist.updateAll()
        cluster = Cluster(neuron)
        s.clusters[cluster.id] = cluster
        neuron.cluster = cluster
        try:
            cw = self.windows['Cluster'] # don't force its display by default
        except KeyError:
            cw = self.OpenWindow('Cluster')
        return cluster

    def DelClusters(self, clusters, update=True):
        """Delete clusters from the GUI, and delete clusters
        and their neurons from the Sort."""
        clusters = toiter(clusters)
        self.SelectClusters(clusters, on=False) # first deselect them all
        sw = self.windows['Sort']
        cw = self.windows['Cluster']
        self.ColourPoints(clusters, setnid=0) # decolour before clusters lose their sids
        for cluster in clusters:
            sw.RemoveNeuron(cluster.neuron, update=update)
        cw.glWidget.updateGL()
        if update:
            self.UpdateClustersGUI()

    def UpdateClustersGUI(self):
        """Update lots of stuff after modifying clusters,
        here as a separate method for speed, only call when really needed"""
        s = self.sort
        sw = self.windows['Sort']
        sw.nlist.updateAll()
        s.update_usids()
        sw.uslist.updateAll()

    def ColourPoints(self, clusters, setnid=None):
        """Colour the points that fall within each cluster (as specified
        by cluster.neuron.sids) the same colour as the cluster itself. Or, if
        setnid != None, colour all points in clusters according to setnid value"""
        clusters = toiter(clusters)
        gw = self.windows['Cluster'].glWidget
        for cluster in clusters:
            neuron = cluster.neuron
            # not all (or any) of neuron.sids may currently be plotted
            commonsids = np.intersect1d(neuron.sids, gw.sids)
            if len(commonsids) > 0:
                sidis = gw.sids.searchsorted(commonsids)
                # set new nids for commonsids in glWidget:
                if setnid == None:
                    gw.nids[sidis] = neuron.id
                else:
                    gw.nids[sidis] = setnid
                gw.colour(commonsids) # recolour commonsids according to their nids
        gw.updateGL()

    def GetClusterPlotDims(self):
        """Return 3-tuple of strings of cluster dimension names, in (x, y, z) order"""
        x = str(self.ui.xDimComboBox.currentText())
        y = str(self.ui.yDimComboBox.currentText())
        z = str(self.ui.zDimComboBox.currentText())
        return x, y, z

    def AddClusterChangeToStack(self, cc):
        """Adds cc to the cluster change stack, removing any potential redo changes"""
        self.cci += 1
        del self.cchanges[self.cci::] # remove any existing redo cluster changes
        self.cchanges.append(cc) # add to stack
        # TODO: check if stack has gotten too long, if so, remove some from the start
        # and update self.cci appropriately

    def ApplyClusterChange(self, cc, direction):
        """Apply cluster change described in cc, in either the forward or backward
        direction, to the current set of clusters"""
        s = self.sort
        spikes = s.spikes
        sw = self.windows['Sort']
        cw = self.windows['Cluster']
        sids = cc.sids

        # reverse meaning of 'new' and 'old' if direction == 'forward', ie if redoing
        if direction == 'back':
            #newnids = cc.newnids # not needed
            oldnids = cc.oldnids
            newunids = cc.newunids
            oldunids = cc.oldunids
            poss = cc.oldposs
            normposs = cc.oldnormposs
            norder = cc.oldnorder
            good = cc.oldgood
        else: # direction == 'forward'
            #newnids = cc.oldnids # not needed
            oldnids = cc.newnids
            newunids = cc.oldunids
            oldunids = cc.newunids
            poss = cc.newposs
            normposs = cc.newnormposs
            norder = cc.newnorder
            good = cc.newgood

        # delete newly added clusters
        newclusters = [ s.clusters[nid] for nid in newunids ]
        self.SelectClusters(newclusters, on=False) # deselect new clusters
        # temporarily deselect any bystander clusters to get around fact that
        # selections are row-based in Qt, not value-based, which means selection
        # changes happen without a selectionChanged event when the rowCount changes
        bystanders = self.GetClusters()
        self.SelectClusters(bystanders, on=False)
        self.DelClusters(newclusters, update=False) # del new clusters

        # restore relevant spike fields
        spikes['nid'][sids] = oldnids

        # restore the old clusters
        oldclusters = []
        dims = self.GetClusterPlotDims()
        t0 = time.time()
        # NOTE: oldunids are not necessarily sorted
        for nid, pos, normpos in zip(oldunids, poss, normposs):
            nsids = sids[oldnids == nid] # sids belonging to this nid
            cluster = self.CreateCluster(update=False, id=nid)
            oldclusters.append(cluster)
            neuron = cluster.neuron
            sw.MoveSpikes2Neuron(nsids, neuron, update=False)
            cluster.pos = pos
            cluster.normpos = normpos
        # restore norder and good
        s.norder = copy(norder)
        s.good = copy(good)

        # now do some final updates
        self.UpdateClustersGUI()
        self.ColourPoints(oldclusters)
        #print('applying clusters to plot took %.3f sec' % (time.time()-t0))
        # select newly recreated oldclusters
        self.SelectClusters(oldclusters)
        # restore bystander selections
        self.SelectClusters(bystanders)
        #print('oldclusters: %r' % [c.id for c in oldclusters])
        #print('newclusters: %r' % [c.id for c in newclusters])
        #print('bystanders: %r' % [c.id for c in bystanders])

    def SplitSpikes(self, delete=False):
        """Split off explicitly selected spikes from their clusters (if any). More accurately,
        split selected cluster(s) into new cluster(s) plus a destination cluster, whose ID
        depends on the delete arg. This process is required to allow undo/redo"""
        oldclusters = self.GetClusters()
        s = self.sort
        spikes = s.spikes
        sids = np.concatenate([self.GetClusterSpikes(), self.GetUnsortedSpikes()])
        sids.sort()
        if len(sids) == 0:
            return # do nothing
        if delete:
            newnid = 0 # junk cluster
        else:
            newnid = s.nextnid
        selsids = self.GetSpikes() # explicitly selected spikes
        selsidis = sids.searchsorted(selsids)
        nids = spikes[sids]['nid'] # seems to return a copy
        nids[selsidis] = newnid # doesn't seem to overwrite nid values in spikes recarray
        self.apply_clustering(oldclusters, sids, nids, verb='split')

    def updateTitle(self):
        """Update main spyke window title based on open stream and sort, if any"""
        if hasattr(self.hpstream, 'fname'):
            title = self.hpstream.fname
            if hasattr(self, 'sort') and self.sort.fname:
                title += ', ' + self.sort.fname
        elif hasattr(self, 'sort') and self.sort.fname:
            title = self.sort.fname
        else:
            title = 'spyke'
        self.setWindowTitle(title) # update the title

    def OpenRecentFile(self):
        """Open a filename from the clicked recent file in the File menu"""
        action = self.sender()
        if action:
            fullfname = qvar2str(action.data())
            self.OpenFile(fullfname)

    def updateRecentFiles(self, fullfname=None):
        """Update list of recent files in File menu, optionally specifying the
        last fname opened or closed, which should hence go to the top of the list.
        Some of this code is taken from PySide's examples/mainwindows/recentfiles.py"""
        settings = QtCore.QSettings('spyke', 'spyke') # retrieve setting
        fullfnames = qvar2list(settings.value('recentFileList'))
        for i in range(len(fullfnames)): # Py2: convert each entry from QVariant to QString
            fullfnames[i] = qvar2str(fullfnames[i])
        if fullfname:
            try:
                fullfnames.remove(fullfname)
            except ValueError:
                pass
            fullfnames.insert(0, fullfname)
        del fullfnames[MAXRECENTFILES:]
        settings.setValue('recentFileList', fullfnames) # update setting

        # update menu to match fullfnames:
        nrecent = len(fullfnames)
        for i, fullfname in enumerate(fullfnames):
            text = "&%d %s" % (i, fullfname) # add keyboard accelerator
            self.recentFileActions[i].setText(text)
            self.recentFileActions[i].setData(fullfname)
            self.recentFileActions[i].setVisible(True)

        for j in range(nrecent, MAXRECENTFILES):
            self.recentFileActions[j].setVisible(False)

    def OpenFile(self, fname):
        """Open a stream or sort file. fname in this case must contain a full path"""
        print('Opening file %r' % fname)
        head, tail = os.path.split(fname)
        assert head # make sure fname has a path to it
        base, ext = os.path.splitext(tail)
        if ext in ['.dat', '.ns6', '.srf', '.track', '.tsf', '.mat']:
            self.streampath = head
            self.OpenStreamFile(tail)
        elif ext == '.zip':
            subext = os.path.splitext(base)[1]
            self.eventspath = head
            if subext == '.eventwaves':
                self.OpenEventWavesFile(tail)
            elif subext == '.events':
                self.OpenEventsFile(tail)
        elif ext == '.sort':
            self.sortpath = head
            self.OpenSortFile(tail)
        else:
            critical = QtGui.QMessageBox.critical
            critical(self, "Error", "%s is not a .dat, .ns6, .srf, .track, .tsf, .mat, "
                                    ".event*.zip or .sort file" % fname)

    def OpenStreamFile(self, fname):
        """Open a stream (.dat, .ns6, .srf, .track, or .tsf file) and update display
        accordingly. fname is assumed to be relative to self.streampath"""
        if self.hpstream is not None:
            self.CloseStream() # in case a stream is already open
        enabledchans = None
        ext = os.path.splitext(fname)[1]
        if ext == '.dat':
            f = dat.File(fname, self.streampath) # parses immediately
            self.hpstream = f.hpstream # highpass record (spike) stream
            self.lpstream = f.lpstream # lowpassmultichan record (LFP) stream
        elif ext == '.ns6':
            f = nsx.File(fname, self.streampath) # parses immediately
            self.hpstream = f.hpstream # highpass record (spike) stream
            self.lpstream = f.lpstream # lowpassmultichan record (LFP) stream
        elif ext == '.srf':
            f = surf.File(fname, self.streampath)
            f.parse() # TODO: parsing progress dialog
            self.hpstream = f.hpstream # highpass record (spike) stream
            self.lpstream = f.lpstream # lowpassmultichan record (LFP) stream
        elif ext == '.track':
            fs = []
            with open(os.path.join(self.streampath, fname), 'r') as trackfile:
                for line in trackfile: # one filename per line
                    line = line.strip() # remove leading and trailing whitespace
                    print('%s' % line)
                    if not line: # blank line
                        continue
                    if line.startswith('#'): # comment line
                        line = lstrip(line, '#') # remove comment character
                        line = line.replace(' ', '') # remove all spaces
                        if line.startswith('enabledchans='):
                            # it's a comment line describing which chans have been set to
                            # enabled for this track
                            enabledchans = np.asarray(eval(lstrip(line, 'enabledchans=')))
                            assert iterable(enabledchans)
                        continue # to next line
                    fn = line
                    fext = os.path.splitext(fn)[1]
                    if fext == '.dat':
                        f = dat.File(fn, self.streampath)
                    elif fext == '.ns6':
                        f = nsx.File(fn, self.streampath)
                    elif fext == '.srf':
                        f = surf.File(fn, self.streampath)
                        f.parse()
                    else:
                        raise ValueError('unknown extension %r' % fext)
                    fs.append(f) # build up list of open and parsed data file objects
            self.hpstream = MultiStream(fs, fname, kind='highpass')
            self.lpstream = MultiStream(fs, fname, kind='lowpass')
            ext = fext # for setting *tw variables below
        elif ext == '.tsf':
            self.hpstream, self.lpstream = self.OpenTSFFile(fname)
        elif ext == '.mat':
            self.hpstream = self.OpenQuirogaMATFile(fname)
            ext = '.srf' # use same *tw variables as for .srf
        else:
            raise ValueError('unknown extension %r' % ext)

        # if a sort is already open, try rebinding new stream to the sort. If they don't match,
        # abort opening of the new stream:
        try:
            self.sort.stream = self.hpstream # restore newly opened stream to sort
        except AttributeError: # no sort yet
            pass
        except ValueError: # from sort.set_stream()
            print('Aborting opening of the stream')
            self.CloseStream()
            raise # re-raise the ValueError from sort.set_stream()

        self.updateTitle()
        self.updateRecentFiles(os.path.join(self.streampath, fname))

        self.ui.__dict__['actionFiltmeth%s' % self.hpstream.filtmeth ].setChecked(True)
        self.ui.__dict__['actionCAR%s' % self.hpstream.car ].setChecked(True)
        try:
            sampfreqkHz = self.hpstream.sampfreq / 1000
            self.ui.__dict__['action%dkHz' % sampfreqkHz].setChecked(True)
        except KeyError:
            print('WARNING: %d kHz is not a sampling menu option' % sampfreqkHz)
        self.ui.actionSampleAndHoldCorrect.setChecked(self.hpstream.shcorrect)

        self.spiketw = SPIKETW[ext] # spike window temporal window (us)
        self.charttw = CHARTTW[ext] # chart window temporal window (us)
        self.lfptw = LFPTW # lfp window temporal window (us)

        self.uVperum = UVPERUM[ext]
        self.usperum = USPERUM[ext]

        self.ui.dynamicNoiseXSpinBox.setValue(DYNAMICNOISEX[ext])
        self.ui.dtSpinBox.setValue(DT[ext])

        # if a .sort file is already open, enable only those channels that were used
        # by the sort's Detector:
        try:
            enabledchans = self.sort.detector.chans
        except AttributeError:
            pass

        if enabledchans is None:
            self.chans_enabled = self.hpstream.chans
        else:
            print('setting enabled chans = %s' % enabledchans)
            self.chans_enabled = enabledchans

        self.trange = self.hpstream.t0, self.hpstream.t1 # us
        self.t = self.trange[0] # init current timepoint (us)
        self.str2t = {'start': self.trange[0],
                      'now'  : self.t,
                      'end'  : self.trange[1]}

        self.SPIKEWINDOWWIDTH = self.hpstream.probe.ncols * SPIKEWINDOWWIDTHPERCOLUMN
        self.OpenWindow('Spike')

        self.ui.filePosLineEdit.setText('%.1f' % self.t)
        self.ui.filePosStartButton.setText('%.1f' % self.trange[0])
        self.ui.filePosEndButton.setText('%.1f' % self.trange[1])
        self.update_slider() # set slider limits and step sizes

        self.EnableStreamWidgets(True)

    def OpenQuirogaMATFile(self, fname):
        """Open Quiroga's .mat files containing single channel synthetic highpass spike data.
        Return a SimpleStream. Assume no sample-and-hold correction is required, and no
        highpass filtering is required"""
        import scipy.io
        fname = os.path.join(self.streampath, fname)
        d = scipy.io.loadmat(fname, squeeze_me=True)
        #chan = d['chan'] # this field isn't always present
        #assert chan == 1
        nchans = 1
        wavedata = d['data'] # float64, mV
        wavedata = wavedata * 1000 # uV
        assert wavedata.ndim == 1
        nt = len(wavedata)
        wavedata.shape = nchans, -1 # enforce 2D
        # convert to int16, assume ADC resolution for this data was <= 16 bits,
        # use some reasonable gain values, check they don't saturate 16 bits:
        intgain = 1
        extgain = 2000
        converter = core.Converter(intgain=intgain, extgain=extgain)
        wavedata = converter.uV2AD(wavedata, dtype=np.int64)
        # check for saturation:
        wdmin, wdmax = wavedata.min(), wavedata.max()
        print('gain = %d' % (intgain*extgain))
        print('wavedata.min() = %d, wavedata.max() = %d' % (wdmin, wdmax))
        if wdmin <= -2**15 or wdmax >= 2**15-1:
            raise RuntimeError("wavedata has saturated int16. Try reducing gain")
        wavedata = np.int16(wavedata) # downcast to int16
        siteloc = np.empty((nchans, 2))
        siteloc[0] = 0, 0
        rawtres = float(d['samplingInterval']) # ms
        rawtres = rawtres / 1000 # sec
        rawsampfreq = intround(1 / rawtres) # Hz
        masterclockfreq = None
        stream = SimpleStream(fname, wavedata, siteloc, rawsampfreq, masterclockfreq,
                              intgain, extgain, shcorrect=False, bitshift=None)
        truth = core.EmptyClass()
        truth.spiketis = d['spike_times']
        assert truth.spiketis[-1] < nt
        truth.spikets = truth.spiketis * rawtres
        # unsure what the other arrays in this field are for:
        truth.sids = d['spike_class'][0]
        assert int(d['startData']) == 0
        stream.truth = truth
        return stream

    def OpenTSFFile(self, fname):
        """Open NVS's "test spike file" .tsf format for testing spike sorting performance.
        This describes a single 2D contiguous array of raw waveform data, within which are
        embedded a number of spikes from a number of neurons. The ground truth is typically
        listed at the end of the file. Return a highpass and lowpass SimpleStream. For .tsf
        files that only have highpass, return None as a lowpass stream.

        fname is assumed to be relative to self.streampath.

        .tsf file TODO:

            - make data column-major for better seeking in time
            - move nchans field before siteloc field
            - make maxchans 0 based, ie same as labelled on probe design by UMich
            - would be better to keep spikes sorted in time, instead of by cluster id
            - no need for 64 extgain values, they're all the same, whether you're exporting
              spike or LFP data. And if for some reason they could've been different, length
              of extgains vector should be nchans, not fixed 64. Also, if extgains is a
              vector, then so should intgains
            - number cluster ids in vertically spatial order, by mean of their template's
              vertical spatial position, not just by their maxchan - subtle difference
            - are .tsf spike times all aligned to +ve 0 crossing? One difference from .sort
              is that they're all truncated to the nearest 25kHz sample point. Maybe it
              would be best to save the spike time in us instead of in 25kHz sample point
              indices
            - add some kind of datetime stamp, ala .srf. Maybe datetime the .tsf file was
              generated
            - increment format number. Maybe we should ultimately make a .nvs file
              type, similar to .tsf format, for sharing with others, as a simplified
              .srf file. Would require adding an LFP channel field to the end, or just make
              the LFP chans look like normal spike chans, way oversampled
            - add more cells, make some fraction of them bursting, give bursting cells
              some prob distrib over number of spikes per burst, make each spike in a
              burst say 5 or 10% smaller than the previous spike adaptation
            - maybe even simulate spatial drift? That would be more difficult
            - need far more spikes. Enforce a power law distribution in number spikes
              per cell
            - main thing is to look at how close in space and time spikes can be seeded
              and still be detected and clustered correctly
    
        """
        with open(os.path.join(self.streampath, fname), 'rb') as f:
            header = f.read(16).decode()
            assert header == 'Test spike file '
            version, = unpack('i', f.read(4))

        if version == 1002:
            return self.OpenTSFFile_1002(fname)
        elif version == 1000:
            return self.OpenTSFFile_1000(fname)

    def OpenTSFFile_1002(self, fname):
        """Open TSF file, version 1002. Assume no sample-and-hold correction is required,
        assume wavedata already has the correct 0 voltage offset (i.e., is signed), assume no
        bitshift is required (data is 16 bit, not 12). Assume wavedata is wideband, containing
        both spike and LFP data"""
        try: f = open(os.path.join(self.streampath, fname), 'rb')
        except IOError:
            print("Can't find file %r" % fname)
            return
        header = f.read(16).decode()
        assert header == 'Test spike file '
        version, = unpack('i', f.read(4))
        assert version == 1002
        rawsampfreq, = unpack('i', f.read(4)) # Hz
        masterclockfreq = None
        nchans, = unpack('i', f.read(4))
        nt, = unpack('i', f.read(4))
        intgain = 1 # assumed
        extgain, = unpack('f', f.read(4))
        print('extgain: %f' % extgain)
        siteloc = np.zeros((nchans, 2), dtype=np.int16)
        readloc = np.zeros(nchans, dtype=np.int32) # optimal chan display order
        #print('readloc:', readloc)
        for i in range(nchans):
            # these two data types really shouldn't be intertwined like this:
            siteloc[i, :] = unpack('hh', f.read(4))
            readloc[i], = unpack('i', f.read(4))
        # read row major data, ie, chan loop is outer loop:
        wavedata = np.fromfile(f, dtype=np.int16, count=nchans*nt)
        wavedata.shape = nchans, nt
        nspikes, = unpack('i', f.read(4))
        print("%d ground truth spikes" % nspikes)
        # filter into highpass data:
        hpwavedata = core.WMLDR(wavedata)
        # assume all 16 bits are actually used, not just 12 bits, so no bitshift is required:
        hpstream = SimpleStream(fname, hpwavedata, siteloc, rawsampfreq, masterclockfreq,
                                intgain, extgain, shcorrect=False, bitshift=False,
                                tsfversion=version)
        lpstream = None ## TODO: implement this
        if nspikes > 0:
            truth = core.EmptyClass()
            truth.spikets = np.fromfile(f, dtype=np.int32, count=nspikes)
            truth.nids = np.fromfile(f, dtype=np.int32, count=nspikes)
            truth.maxchans = np.fromfile(f, dtype=np.int32, count=nspikes)
            assert truth.maxchans.min() >= 1 # NVS stores these as 1-based
            truth.maxchans -= 1 # convert to proper 0-based maxchan ids
            self.renumber_tsf_truth(truth, hpstream)
            hpstream.truth = truth
        pos = f.tell()
        f.seek(0, 2)
        nbytes = f.tell()
        f.close()
        print('Read %d bytes, %s is %d bytes long' % (pos, fname, nbytes))
        return hpstream, lpstream

    def OpenTSFFile_1000(self, fname):
        """Open TSF file, version 1000. Assume wavedata is highpass spike data only"""
        try: f = open(os.path.join(self.streampath, fname), 'rb')
        except IOError:
            print("Can't find file %r" % fname)
            return
        header = f.read(16).decode()
        assert header == 'Test spike file '
        version, = unpack('i', f.read(4))
        assert version == 1000
        nchans = 54 # assumed
        siteloc = np.fromfile(f, dtype=np.int16, count=nchans*2)
        siteloc.shape = nchans, 2
        rawsampfreq, = unpack('i', f.read(4)) # 25k
        masterclockfreq, = unpack('i', f.read(4)) # 1M
        extgains = np.fromfile(f, dtype=np.uint16, count=64)
        extgain = extgains[0]
        intgain, = unpack('H', f.read(2))
        # this nchans field should've been above siteloc field:
        nchans2, = unpack('i', f.read(4))
        assert nchans == nchans2 # make sure above assumption was right
        nt, = unpack('i', f.read(4)) # 7.5M, eq'v to 300 sec data total
        # read row major data, ie, chan loop is outer loop:
        wavedata = np.fromfile(f, dtype=np.int16, count=nchans*nt)
        wavedata.shape = nchans, nt
        hpstream = SimpleStream(fname, wavedata, siteloc, rawsampfreq, masterclockfreq,
                                intgain, extgain, shcorrect=True, tsfversion=version)
        lpstream = None # no lowpass data in this version
        # not all .tsf files have ground truth data at end:
        pos = f.tell()
        groundtruth = f.read()
        if groundtruth == b'': # reached EOF
            nbytes = f.tell()
            f.close()
            print('Read %d bytes, %s is %d bytes long' % (pos, fname, nbytes))
            return hpstream, lpstream
        else:
            f.seek(pos) # go back and parse ground truth data
        truth = core.EmptyClass()
        # something to do with how spikes were seeded vertically in space:
        truth.vspacing, = unpack('i', f.read(4))
        truth.nspikes, = unpack('i', f.read(4))
        # sample index of each spike:
        spiketis = np.fromfile(f, dtype=np.uint32, count=truth.nspikes)
        sids = spiketis.argsort() # indices that sort spikes in time
        truth.spikets = spiketis[sids] * hpstream.rawtres # in us
        truth.nids = np.fromfile(f, dtype=np.uint32, count=truth.nspikes)[sids]
        truth.chans = np.fromfile(f, dtype=np.uint32, count=truth.nspikes)[sids]
        assert truth.chans.min() >= 1 # NVS stores these as 1-based
        truth.chans -= 1 # convert to proper 0-based maxchan ids
        self.renumber_tsf_truth(truth, hpstream)
        hpstream.truth = truth
        pos = f.tell()
        f.seek(0, 2)
        nbytes = f.tell()
        f.close()
        print('Read %d bytes, %s is %d bytes long' % (pos, fname, nbytes))
        return hpstream, lpstream

    def renumber_tsf_truth(self, truth, stream):
        """Renumber .tsf ground truth nids according to vertical spatial order of their
        max chan, similar to what's done in .sort. Differences in labelling can still
        arise because in a .sort, nids are ordered by the mean vertically modelled
        position of each neuron's member spikes, not strictly by the maxchan of its
        mean template"""
        oldnid2sids = {}
        nids = truth.nids
        oldunids = np.unique(nids)
        nnids = len(oldunids)
        oldchans = np.zeros(nnids, dtype=truth.chans.dtype)
        assert (oldunids == np.arange(1, nnids+1)).all()
        # find maxchan of each nid, store in oldchans:
        for chani, oldnid in enumerate(oldunids):
            sids = nids == oldnid
            oldnid2sids[oldnid] = sids # save these for next loop
            chans = truth.chans[sids]
            chan = chans[0]
            assert (chans == chan).all() # check for surprises
            oldchans[chani] = chan
        # convert maxchans to y positions:
        ypos = np.asarray([ stream.probe.SiteLoc[chan][1] for chan in oldchans ])
        # as in sort.on_actionRenumberClusters_triggered(), this is a bit confusing:
        # find indices that would sort old ids by y pos, but then what you really want
        # is to find the y pos *rank* of each old id, so you need to take argsort again:
        sortiis = ypos.argsort().argsort()
        newunids = oldunids[sortiis] # sorted by vertical position
        for oldnid, newnid in zip(oldunids, newunids):
            sids = oldnid2sids[oldnid]
            nids[sids] = newnid # overwrite old nid values with new ones

    def OpenEventWavesFile(self, fname):
        """Open and import the data in an .eventwaves.zip file, containing event times,
        channels and waveforms, plus some other data. fname is assumed to be relative to
        self.eventspath"""
        if self.hpstream != None:
            self.CloseStream() # in case a stream is open
        self.DeleteSort() # delete any existing Sort
        fullfname = os.path.join(self.eventspath, fname)
        with open(fullfname, 'rb') as f:
            d = dict(np.load(f)) # convert to an actual dict to use d.get() method
            print('Done opening .eventswave.zip file')
            print('.eventswave.zip file was %d bytes long' % f.tell())
            chan = d.get('chan') # array of maxchans, one per event
            chanpos = d.get('chanpos') # array of (x, y) coords, in channel order
            chans = d.get('chans') # set of incl. chans, each of length nchans, one per event
            nchans = d.get('nchans') # count of included chans, one per event
            sampfreq = d.get('sampfreq') # sampling rate, Hz
            t = d.get('t') # even timestamps, us
            uVperAD = d.get('uVperAD') # uV per AD value in wavedata
            # event waveform data (nevents x maxnchans x nt), treated as AD values:
            wavedata = d.get('wavedata')

        # check for mandatory fields:
        if sampfreq is None:
            raise ValueError('missing sampfreq')
        if uVperAD is None:
            raise ValueError('missing uVperAD')
        if wavedata is None:
            raise ValueError('missing wavedata')

        # pull singleton values out of numpy array:
        sampfreq = float(sampfreq)
        uVperAD = float(uVperAD)

        nevents, maxnchans, nt = wavedata.shape # maxnchans is per event
        print('wavedata.shape:', wavedata.shape)

        # handle optional fields:
        if chanpos is None:
            if maxnchans > 1:
                raise ValueError('multiple chans per event, chanpos should be specified')
            chanpos = np.array([[0, 0]]) # treat events as single channel
        if t is None: # create artificial event timestamps at 1 ms intervals
            t = np.arange(nevents) * 1000 # us
        if chan is None: # maxchan
            chan = np.zeros(nevents)
        if nchans is None:
            nchans = np.ones(nevents)
        if chans is None:
            chans = np.asarray([chan]) # (1, nevents)
        assert len(chans) is maxnchans

        # create fake stream, create sort, populate spikes array:
        tres = 1 / sampfreq * 1000000 # us
        halfdt = nt * tres / 2
        self.spiketw = -halfdt, halfdt

        # treat this source .eventwaves.zip file as a fake stream:
        fakestream = stream.FakeStream()
        fakestream.fname = fname
        fakestream.tres = tres
        fakestream.probe = probes.findprobe(chanpos)
        fakestream.converter = None
        self.hpstream = fakestream

        sort = self.CreateNewSort() # create a new sort, with bound stream
        det = Detector(sort=sort)
        SPIKEDTYPE = calc_SPIKEDTYPE(maxnchans)
        sort.detector = det
        sort.converter = core.SimpleConverter(uVperAD)
        spikes = np.zeros(nevents, SPIKEDTYPE)
        spikes['id'] = np.arange(nevents)
        spikes['t'] = t
        spikes['t0'], spikes['t1'] = t-halfdt, t+halfdt
        spikes['chan'] = chan
        spikes['nchans'] = nchans
        spikes['chans'] = chans.T # (nevents, 1)
        sort.spikes = spikes
        sort.wavedata = wavedata

        # hack:
        self.uVperum = 20
        self.usperum = 125

        sort.update_usids() # required for self.on_plotButton_clicked()

        # lock down filtmeth, car, sampfreq and shcorrect attribs:
        #sort.filtmeth = sort.stream.filtmeth
        #sort.car = sort.stream.car
        #sort.sampfreq = sort.stream.sampfreq
        #sort.shcorrect = sort.stream.shcorrect

        self.ui.progressBar.setFormat("%d spikes" % sort.nspikes)
        self.EnableSortWidgets(True)
        sw = self.OpenWindow('Sort') # ensure it's open
        if sort.nspikes > 0:
            self.on_plotButton_clicked()

        self.SPIKEWINDOWWIDTH = sort.probe.ncols * SPIKEWINDOWWIDTHPERCOLUMN
        self.updateTitle()
        self.updateRecentFiles(fullfname)

        # start with all events in a single non-junk cluster 1:
        oldclusters = []
        sids = spikes['id']
        nids = np.ones(nevents)
        self.apply_clustering(oldclusters, sids, nids, verb='initial eventwaves split')

    def OpenEventsFile(self, fname):
        """Open and import the data in an .events.zip file, containing spike times, channels,
        and neuron ids. fname is assumed to be relative to self.eventspath. Spike waveforms
        are extracted from the currently open stream"""
        if self.hpstream is None:
            raise RuntimeError("Need an open raw data stream before loading an events.zip "
                               "file")
        self.DeleteSort() # delete any existing Sort
        fullfname = os.path.join(self.eventspath, fname)
        with open(fullfname, 'rb') as f:
            d = dict(np.load(f)) # convert to an actual dict to use d.get() method
            print('Done opening .events.zip file')
            print('.events.zip file was %d bytes long' % f.tell())
            spikets = d.get('spikets') # spike times, us
            maxchans = d.get('maxchans') # maxchans
            nids = d.get('nids') # neuron IDs

        # check for mandatory fields:
        if spikets is None:
            raise ValueError('missing spikets')
        if maxchans is None:
            raise ValueError('missing maxchans')
        if nids is None:
            raise ValueError('missing nids')
        assert len(spikets) == len(maxchans) == len(nids)
        nspikes = len(spikets)

        # check that maxchans are a subset of enabled chans in stream:
        umaxchans = np.unique(maxchans)
        if not np.isin(umaxchans, self.hpstream.chans).all():
            raise RuntimeError("maxchans in %r are not a subset of currently enabled stream "
                               "chans. Was the .events.zip file generated from a different "
                               "set of enabled channels?\n"
                               "maxchans: %s\n"
                               "enabled chans: %s\n"
                               % (fname, umaxchans, self.hpstream.chans))

        # create sort:
        print('Creating new sort')
        sort = self.CreateNewSort() # create a new sort, with bound stream
        # create detector and run Detector.predetect(), so that things initialize:
        self.get_detector()
        det = sort.detector
        assert det.extractparamsondetect == True
        self.init_extractor() # init the Extractor
        det.predetect(logpath=self.eventspath)

        # manually set detection results:
        print('Allocating and filling spikes array')
        spikes = np.zeros(nspikes, det.SPIKEDTYPE)
        spikes['id'] = np.arange(nspikes)
        spikes['t'] = spikets
        spikes['t0'], spikes['t1'] = spikets+sort.tw[0], spikets+sort.tw[1]
        spikes['chan'] = maxchans # one maxchan per spike
        # convert inclnbhdi to inclnbhd, taking chan and returning inclchans instead of taking
        # chani and returning inclchanis:
        inclnbhd = {}
        for chani, inclchanis in det.inclnbhdi.items():
            chan = det.chans[chani]
            inclchans = det.chans[inclchanis]
            inclnbhd[chan] = inclchans
        for s, maxchan in zip(spikes, maxchans):
            inclchans = inclnbhd[maxchan]
            nchans = len(inclchans)
            s['nchans'] = nchans
            s['chans'][:nchans] = inclchans
            s['chani'], = np.where(inclchans == maxchan) # index into spike's chan list

        # bind to self:
        sort.spikes = spikes
        det.nspikes = nspikes

        # init wavedata:
        print('Allocating wavedata array')
        sort.wavedata = np.zeros((nspikes, det.maxnchansperspike, det.maxnt), dtype=np.int16)
        # Linux has lazy physical memory allocation. See https://stackoverflow.com/a/27582592.
        # This forces physical memory allocation, though strangely, doesn't seem to speed
        # up loading of wavedata. It will fail immediately if physical memory can't be
        # allocated, which is desirable:
        sort.wavedata[:] = 0
        print('wavedata.shape:', sort.wavedata.shape)
        print('wavedata.nbytes: %.3f GiB' % (sort.wavedata.nbytes / 1024**3))
        # "re"load spike wavedata based on imported events:
        sort.reload_spikes(spikes['id'])

        sort.update_usids() # required for self.on_plotButton_clicked()

        # lock down filtmeth, car, sampfreq and shcorrect attribs:
        sort.filtmeth = sort.stream.filtmeth
        sort.car = sort.stream.car
        sort.sampfreq = sort.stream.sampfreq
        sort.shcorrect = sort.stream.shcorrect

        self.ui.progressBar.setFormat("%d spikes" % sort.nspikes)
        self.EnableSortWidgets(True)
        sw = self.OpenWindow('Sort') # ensure it's open
        if sort.nspikes > 0:
            self.on_plotButton_clicked()

        self.SPIKEWINDOWWIDTH = sort.probe.ncols * SPIKEWINDOWWIDTHPERCOLUMN
        self.updateTitle()
        self.updateRecentFiles(fullfname)

        # set nids using apply_clustering():
        oldclusters = []
        sids = spikes['id']
        self.apply_clustering(oldclusters, sids, nids, verb='initial .events.zip split')
        # no longer valid, loaded nids may have had gaps that were removed by
        # apply_clustering():
        del nids

        # now that wavedata have been extracted and neuron mean waveforms calculated,
        # find tis and do spatial localization of each spike:
        ntis, nalignis = {}, {} # tis and aligni derived from each neuron's mean waveform
        for neuron in sort.neurons.values():
            nwave = neuron.get_wave() # update and return mean waveform
            mintis = nwave.data.argmin(axis=1)
            maxtis = nwave.data.argmax(axis=1)
            ntis[neuron.id] = np.column_stack([mintis, maxtis])
            # choose aligni with least variance:
            nalignis[neuron.id] = np.argmin([mintis.std(), maxtis.std()])
        AD2uV = sort.converter.AD2uV
        weights2f = sort.extractor.weights2spatial
        weights2spatialmean = sort.extractor.weights2spatialmean
        f = sort.extractor.f
        nreject = 0 # number spikes rejected during spatial localization
        print('Running spatial localization on all %d spikes' % nspikes)
        for s, wd in zip(sort.spikes, sort.wavedata):
            # Get Vpp at each inclchan's tis, use as spatial weights:
            # see core.rowtake() or util.rowtake_cy() for indexing explanation:
            sid = s['id']
            # print out progress on a regular basis:
            if sid % 10000 == 0:
                printflush(sid, end='')
            elif sid % 1000 == 0:
                printflush('.', end='')
            spiket = intround(s['t']) # nearest us
            nid = s['nid']
            chan = s['chan']
            nchans = s['nchans']
            chans = s['chans'][:nchans]
            neuronchans = sort.neurons[nid].wave.chans
            assert (chans == neuronchans).all()
            s['tis'][:nchans] = ntis[nid] # set according to its neuron, wrt t0i=0
            s['aligni'] = nalignis[nid] # set according to its neuron
            maxchani = s['chani']
            t0i, t1i = int(s['tis'][maxchani, 0]), int(s['tis'][maxchani, 1])
            s['dt'] = abs(t1i - t0i) / sort.sampfreq * 1e6 # us
            # note that V0 and V1 might not be of opposite sign, because tis are derived
            # from mean neuron waveform, not from each individual spike:
            s['V0'], s['V1'] = AD2uV(wd[maxchani, t0i]), wd[maxchani, t1i] # uV
            s['Vpp'] = abs(s['V1'] - s['V0']) # uV
            chanis = det.chans.searchsorted(chans)
            w = np.float32(wd[np.arange(s['nchans'])[:, None], s['tis'][:nchans]]) # nchans x 2
            w = abs(w).sum(axis=1) # Vpp for each chan, measured at t0i and t1i
            x = det.siteloc[chanis, 0] # 1D array (row)
            y = det.siteloc[chanis, 1]
            params = weights2f(f, w, x, y, maxchani)
            if params == None: # presumably a non-localizable many-channel noise event
                #printflush('X', end='') # to indicate a rejected spikes
                if DEBUG: det.log("Reject spike %d at t=%d based on fit params"
                                  % (sid, spiket))
                neuron = sort.neurons[nid]
                # remove from its neuron, add to unsorted list of spikes:
                sw.MoveSpikes2List(neuron, [sid], update=False)
                # manually set localization params to Vpp-weighted spatial mean and 0 sigma:
                x0, y0 = weights2spatialmean(w, x, y)
                # set sigma to 0 um, and then later round lockr up to 1 um so that only one
                # raster tick shows up for each rejected spike, reducing clutter
                params = x0, y0, 0, 0
                nreject += 1
            # Save spatial fit params, and "lockout" only the channels within lockrx*sx
            # of the fit spatial location of the spike, up to a max of inclr. "Lockout"
            # in this case only refers to which channels are highlighted with a raster tick
            # for each spike:
            s['x0'], s['y0'], s['sx'], s['sy'] = params
            x0, y0 = s['x0'], s['y0']
            # lockout radius for this spike:
            lockr = min(det.lockrx*s['sx'], det.inclr) # in um
            lockr = max(lockr, 1) # at least 1 um, so at least the maxchan gets a tick
            # test y coords of chans in y array, ylockchaniis can be used to index
            # into x, y and chans:
            ylockchaniis, = np.where(np.abs(y - y0) <= lockr) # convert bool arr to int
            # test Euclid distance from x0, y0 for each ylockchani:
            lockchaniis = ylockchaniis.copy()
            for ylockchanii in ylockchaniis:
                if dist((x[ylockchanii], y[ylockchanii]), (x0, y0)) > lockr:
                    # Euclidean distance is too great, remove ylockchanii from lockchaniis:
                    lockchaniis = lockchaniis[lockchaniis != ylockchanii]
            lockchans = chans[lockchaniis]
            nlockchans = len(lockchans)
            s['lockchans'][:nlockchans], s['nlockchans'] = lockchans, nlockchans

        print() # newline
        preject = nreject / nspikes * 100
        print('Rejected %d/%d spikes (%.1f %%), set as unclustered'
              % (nreject, nspikes, preject))

        # remove any empty neurons due to all their spikes being rejected:
        nneurons, nnreject = len(sort.neurons), 0
        for neuron in sort.neurons.values():
            if len(neuron.sids) == 0:
                sw.RemoveNeuron(neuron, update=False)
                nnreject += 1
        preject = nnreject / nneurons * 100
        print('Removed %d/%d (%.1f %%) empty neurons'
              % (nnreject, nneurons, preject))

        self.UpdateClustersGUI()

        # update mean cluster positions, so they can be sorted by y0:
        for cluster in sort.clusters.values():
            cluster.update_pos()

        print('Done importing events from %r' % fullfname)

    def convert_kilosortnpy2eventszip(self, path):
        """Read relevant KiloSort .npy results files in path, process them slightly,
        and save them with standard spyke variable names to an ".events.zip" npz file.
        KiloSort .npy results are assumed to correspond to currently open stream."""
        s = self.hpstream
        assert s != None

        # build file names:
        spiketisfname = os.path.join(path, 'spike_times.npy')
        nidsfname = os.path.join(path, 'spike_clusters.npy')
        templatesfname = os.path.join(path, 'templates.npy')
        outputfname = os.path.join(path, s.fname + '.events.zip')
        print('Converting KiloSort events to:\n%r' % outputfname)

        # load relevant KiloSort .npy results files:
        # spike times, sample point integers relative to start of .dat file:
        spiketis = np.load(spiketisfname).ravel()
        nids = np.load(nidsfname).ravel() # 0-based neuron IDs, one per spike
        templates = np.load(templatesfname) # ntemplates, nt, nchans, Fortran contiguous
        # reshape to ntemplates, nchans, nt by swapping axes (can't just assign new shape!):
        templates = np.swapaxes(templates, 1, 2)
        templates = np.ascontiguousarray(templates) # make C contiguous
        ntemplates, nchans, nt = templates.shape
        if nchans != s.nchans:
            raise RuntimeError("Number of chans in 'templates.npy' (%d) doesn't match "
                               "number of currently enabled chans in stream (%d)"
                               % (nchans, s.nchans))

        # calculate spike times to nearest int64 us, assume KiloSort was run on
        # raw uninterpolated data:
        spikets = intround(s.t0 + spiketis / s.rawsampfreq * 1e6) # us

        # shift KiloSort spike times:
        print('Shifting KiloSort spike times by %d us for better positioning in sort window'
              % KILOSORTSHIFTCORRECT)
        spikets = spikets + KILOSORTSHIFTCORRECT

        # find maxchan for each template: find max along time axis of each chan of each
        # template, then find argmax along chan axis of each template:
        templatemaxchanis = abs(templates).max(axis=2).argmax(axis=1) # one per template
        # get dereferenced maxchan IDs, for example, A1x32 probe has 1-based chans, at
        # least when recorded with Blackrock NSP. s.chans are *enabled* chans only:
        templatemaxchans = s.chans[templatemaxchanis] # one per template
        maxchans = templatemaxchans[nids] # one per spike

        # check limits, convert maxchans to uint8:
        assert maxchans.min() >= np.iinfo(np.uint8).min
        assert maxchans.max() <= np.iinfo(np.uint8).max
        maxchans = np.uint8(maxchans) # save space, use same dtype as in SPIKEDTYPE

        # convert to 1-based neuron IDs, reserve 0 for unclustered spikes. Note that
        # KiloSort's 0-based neuron IDs might have gaps, i.e., they don't necessarily span
        # the range 0..nneurons-1:
        nids += 1
        # check limits, convert nids to int16:
        assert nids.min() >= np.iinfo(np.int16).min
        assert nids.max() <= np.iinfo(np.int16).max
        nids = np.int16(nids) # save space, use same dtype as in SPIKEDTYPE

        assert len(spikets) == len(maxchans) == len(nids)
        with open(outputfname, 'wb') as f:
            np.savez_compressed(f, spikets=spikets, maxchans=maxchans, nids=nids)
        print('Done converting KiloSort events')

    def OpenSortFile(self, fname):
        """Open a Sort from a .sort and .spike and .wave file with the same base name,
        restore the stream"""
        self.DeleteSort() # delete any existing Sort
        print('Opening sort file %r' % fname)
        t0 = time.time()
        f = open(os.path.join(self.sortpath, fname), 'rb')
        unpickler = core.SpykeUnpickler(f)
        sort = unpickler.load()
        print('Done opening sort file, took %.3f sec' % (time.time()-t0))
        print('Sort file was %d bytes long' % f.tell())
        f.close()
        sort.fname = fname # update in case file was renamed
        self.sort = sort

        # if a stream is already open, try rebinding it to the sort. If they don't match,
        # abort opening of the sort:
        if self.hpstream != None:
            try:
                sort.stream = self.hpstream # restore open stream to sort
            except AssertionError: # from sort.set_stream()
                self.DeleteSort() # delete the non-matching sort
                raise RuntimeError("Open stream doesn't match the one specified in sort")
        else: # no open stream, need to set uVperum and usperum according to sort type:
            ext = sort.stream.ext
            self.uVperum = UVPERUM[ext]
            self.usperum = USPERUM[ext]

        basefname = os.path.splitext(fname)[0]
        # load .spike file of the same base name:
        sort.spikefname = basefname + '.spike' # update in case of renamed basefname
        self.OpenSpikeFile(sort.spikefname)

        # load .wave file of the same base name:
        sort.wavefname = basefname + '.wave' # update in case of renamed basefname
        sort.wavedata = self.OpenWaveFile(sort.wavefname)

        # try auto-updating sort to latest version:
        if float(sort.__version__) < float(__version__):
            self.update_sort_version()
        
        # restore Sort's tw to self and to spike and sort windows, if applicable:
        #print('sort.tw is %r' % (sort.tw,))
        self.update_spiketw(sort.tw)
        # restore filtering method:
        self.SetFiltmeth(sort.filtmeth)
        # restore CAR method:
        self.SetCAR(sort.car)
        # restore sampfreq and shcorrect:
        self.SetSampfreq(sort.sampfreq)
        self.SetSHCorrect(sort.shcorrect)
        self.ui.progressBar.setFormat("%d spikes" % sort.nspikes)

        self.SPIKEWINDOWWIDTH = sort.probe.ncols * SPIKEWINDOWWIDTHPERCOLUMN
        sw = self.OpenWindow('Sort') # ensure it's open
        sw.uslist.updateAll() # restore unsorted spike listview
        self.restore_clustering_selections()
        self.RestoreClusters2GUI()
        self.updateTitle()
        self.updateRecentFiles(os.path.join(self.sortpath, fname))
        self.update_gui_from_sort()
        self.EnableSortWidgets(True)

    @property
    def has_sort(self):
        """Convenient way of checking if sort exists"""
        try:
            self.sort
            return True
        except AttributeError:
            return False

    def restore_clustering_selections(self):
        """Restore state of last user-selected clustering parameters, specifically those
        that are otherwise not bound to the sort outside of saving it to file. Performs
        reverse of save_clustering_selections()"""
        s = self.sort
        sw = self.OpenWindow('Sort')
        cw = self.OpenWindow('Cluster')
        # try and restore saved component analysis selection:
        try:
            i = self.ui.componentAnalysisComboBox.findText(s.selCA)
            self.ui.componentAnalysisComboBox.setCurrentIndex(i)
        except AttributeError: pass # wasn't saved, loading from old .sort file
        # try and restore saved cluster selection:
        try: self.SelectClusters(s.selnids)
        except AttributeError: pass # wasn't saved, loading from old .sort file
        # try and restore saved sort window channel selection, and manual selection flag:
        try:
            sw.panel.chans_selected = s.selchans
            sw.panel.manual_selection = s.selchansmanual
            # don't save x, y, z dimension selection, leave it at default xyVpp
            # for maximum speed when loading sort file
        except AttributeError: pass # wasn't saved, loading from old .sort file
        # try and restore saved inclt selection:
        try:
            i = sw.incltComboBox.findText(s.selinclt)
            sw.incltComboBox.setCurrentIndex(i)
        except AttributeError: pass # wasn't saved, loading from old .sort file
        # try and restore saved npcsperchan selection:
        try:
            sw.nPCsPerChanSpinBox.setValue(s.npcsperchan)
        except AttributeError: pass # wasn't saved, loading from old .sort file

        sw.panel.update_selvrefs()
        sw.panel.draw_refs() # update

        self.on_plotButton_clicked() # create glyph on first open
        # try and restore saved camera view
        try: cw.glWidget.MV, cw.glWidget.focus = s.MV, s.focus
        except AttributeError: pass

    def OpenSpikeFile(self, fname):
        """Open a .spike file, assign its contents to the spikes array, update dependencies"""
        sort = self.sort
        print('Loading spike file %r' % fname)
        t0 = time.time()
        f = open(os.path.join(self.sortpath, fname), 'rb')
        spikes = np.load(f)
        print('Done opening spike file, took %.3f sec' % (time.time()-t0))
        print('Spike file was %d bytes long' % f.tell())
        f.close()
        sort.spikes = spikes
        # when loading a spike file, make sure the nid field is overwritten
        # in the spikes array. The nids in sort.neurons are always the definitive ones:
        for neuron in sort.neurons.values():
            spikes['nid'][neuron.sids] = neuron.id
        sort.update_usids()

    def OpenWaveFile(self, fname):
        """Open a .wave file and return wavedata array"""
        sort = self.sort
        print('Opening wave file %r' % fname)
        t0 = time.time()
        f = open(os.path.join(self.sortpath, fname), 'rb')
        try:
            del sort.wavedata
            #gc.collect() # ensure memory is freed up to prepare for new wavedata, necessary?
        except AttributeError: pass
        wavedata = np.load(f)
        print('Done opening wave file, took %.3f sec' % (time.time()-t0))
        print('Wave file was %d bytes long' % f.tell())
        f.close()
        if len(wavedata) != sort.nspikes:
            critical = QtGui.QMessageBox.critical
            critical(self, "Error",
                     ".wave file has a different number of spikes from the current Sort")
            raise RuntimeError
        return wavedata

    def CreateNewSort(self):
        """Create a new Sort, bind it to self, and return it"""
        self.DeleteSort()
        self.sort = Sort(detector=None, # detector is assigned in on_detectButton_clicked
                         stream=self.hpstream,
                         tw=self.spiketw)
        self.EnableSortWidgets(True)
        return self.sort

    def SaveSortFile(self, fname):
        """Save sort to a .sort file. fname is assumed to be relative to self.sortpath"""
        s = self.sort
        try: s.spikes
        except AttributeError: raise RuntimeError("Sort has no spikes to save")
        if not os.path.splitext(fname)[1]: # if it doesn't have an extension
            fname = fname + '.sort'
        try: s.spikefname
        except AttributeError: # corresponding .spike filename hasn't been generated yet
            s.spikefname = os.path.splitext(fname)[0] + '.spike'
        self.SaveSpikeFile(s.spikefname) # always (re)save .spike when saving .sort
        print('Saving sort file %r' % fname)
        t0 = time.time()
        self.save_clustering_selections()
        self.save_window_states()
        s.fname = fname # bind it now that it's about to be saved
        f = open(os.path.join(self.sortpath, fname), 'wb')
        pickle.dump(s, f, protocol=-1) # pickle with most efficient protocol
        f.close()
        print('Done saving sort file, took %.3f sec' % (time.time()-t0))
        self.updateTitle()
        self.updateRecentFiles(os.path.join(self.sortpath, fname))

    def save_clustering_selections(self):
        """Save state of last user-selected clustering parameters. Unlike parameters such as
        sort.sigma, these parameters aren't bound to the sort during normal operation
        yet they're useful to restore when .sort file is reopened"""
        s = self.sort
        sw = self.windows['Sort'] # should be open if s.spikes exists
        s.selCA = str(self.ui.componentAnalysisComboBox.currentText())
        s.selnids = self.GetClusterIDs() # save current cluster selection
        s.selchans = sw.panel.chans_selected
        s.selchansmanual = sw.panel.manual_selection
        s.selinclt = str(sw.incltComboBox.currentText())
        try:
            cw = self.windows['Cluster']
            s.MV, s.focus = cw.glWidget.MV, cw.glWidget.focus # save camera view
        except KeyError:
            # cw hasn't been opened yet, no camera view to save
            pass

    def save_window_states(self):
        """Save window geometries and states (toolbar positions, etc.) to .sort file"""
        s = self.sort
        s.windowGeometries = {}
        s.windowStates = {}
        for wintype, window in self.windows.items():
            #print('saving state of %s window' % wintype)
            s.windowGeometries[wintype] = window.saveGeometry()
            s.windowStates[wintype] = window.saveState()

    def SaveSpikeFile(self, fname):
        """Save spikes to a .spike file. fname is assumed to be relative to self.sortpath"""
        s = self.sort
        try: s.spikes
        except AttributeError: raise RuntimeError("Sort has no spikes to save")
        if not os.path.splitext(fname)[1]: # if it doesn't have an extension
            fname = fname + '.spike'
        try: s.wavefname
        except AttributeError: # corresponding .wave file hasn't been created yet
            wavefname = os.path.splitext(fname)[0] + '.wave'
            # only write whole .wave file if missing s.wavefname attrib:
            self.SaveWaveFile(wavefname)
            self.dirtysids.clear() # shouldn't be any, but clear anyway just in case
        if len(self.dirtysids) > 0:
            self.SaveWaveFile(s.wavefname, sids=self.dirtysids)
            self.dirtysids.clear() # no longer dirty
        print('Saving spike file %r' % fname)
        t0 = time.time()
        f = open(os.path.join(self.sortpath, fname), 'wb')
        np.save(f, s.spikes)
        f.close()
        print('Done saving spike file, took %.3f sec' % (time.time()-t0))
        s.spikefname = fname # used to indicate that the spikes have been saved

    def SaveWaveFile(self, fname, sids=None):
        """Save waveform data to a .wave file. Optionally, update only sids
        in existing .wave file. fname is assumed to be relative to self.sortpath"""
        s = self.sort
        try: s.wavedata
        except AttributeError: return # no wavedata to save
        if not os.path.splitext(fname)[1]: # if it doesn't have an extension
            fname = fname + '.wave'
        print('Saving wave file %r' % fname)
        t0 = time.time()
        if sids is not None and len(sids) >= NDIRTYSIDSTHRESH:
            sids = None # resave all of them for speed
        if sids is None: # write the whole file
            print('Updating all %d spikes in wave file %r' % (s.nspikes, fname))
            f = open(os.path.join(self.sortpath, fname), 'wb')
            np.save(f, s.wavedata)
            f.close()
        else: # write only sids
            print('Updating %d spikes in wave file %r' % (len(sids), fname))
            core.updatenpyfilerows(os.path.join(self.sortpath, fname), sids, s.wavedata)
        print('Done saving wave file, took %.3f sec' % (time.time()-t0))
        s.wavefname = fname

    def DeleteSort(self):
        """Delete any existing Sort"""
        try:
            # TODO: if Save button is enabled, check if Sort is saved,
            # if not, prompt to save
            #print('Deleting existing Sort and entries in list controls')
            #self.sort.spikes.resize(0, recheck=False) # doesn't work, doesn't own memory
            del self.sort
        except AttributeError:
            pass
        if 'Sort' in self.windows:
            sw = self.windows['Sort']
            sw.nlist.reset()
            sw.nslist.reset()
            sw.nslist.neurons = []
            sw.uslist.reset()
            sw.panel.removeAllItems()
            self.HideWindow('Sort')
        if 'Cluster' in self.windows:
            cw = self.windows['Cluster']
            cw.glWidget.reset()
            self.HideWindow('Cluster')
        if 'MPL' in self.windows:
            mplw = self.windows['MPL']
            mplw.ax.clear()
            mplw.figurecanvas.draw()
            self.HideWindow('MPL')
        del self.cchanges[:]
        self.cci = -1
        self.ui.progressBar.setFormat('0 spikes')
        # make sure self.sort and especially self.sort.spikes is really gone
        # TODO: check if this is necessary once everything works with new streamlined
        # (no objects) spikes struct array
        gc.collect()

    def get_chans_enabled(self):
        return self.hpstream.chans

    def set_chans_enabled(self, chans):
        """Updates chans in the streams and plot panels"""
        # update streams:
        self.hpstream.chans = chans
        if self.lpstream.ext == '.srf': # a Surf-like lpstream with a .layout attrib
            # take intersection of lpstream.layout.chans and chans,
            # conserving ordering in lpstream.layout.chans
            self.lpstream.chans = np.asarray([ chan for chan in self.lpstream.layout.chans if
                                               chan in chans ])
        else: # treat it the same as an hpstream
            self.lpstream.chans = chans

        # set chans in plotpanels to reset colours:
        for wintype in WINDOWUPDATEORDER:
            try:
                self.windows[wintype].panel.set_chans(chans)
            except KeyError: # wintype hasn't been opened yet
                pass
        self.plot() # replot

    chans_enabled = property(get_chans_enabled, set_chans_enabled)

    def CloseStream(self):
        """Close data windows and stream (both hpstream and lpstream).
        Caller should first check if there are any streams to close"""
        # need to specifically get a list of keys, not an iterator,
        # since self.windows dict changes size during iteration
        for wintype in list(self.windows): # get keys as list before modifying dict
            if wintype in ['Spike', 'Chart', 'LFP']:
                self.CloseWindow(wintype) # deletes from dict
        for stream in [self.hpstream, self.lpstream]:
            if stream: stream.close()
        self.hpstream = None
        self.lpstream = None
        self.t = None
        self.ShowRasters(False) # reset
        self.updateTitle()
        self.EnableStreamWidgets(False)
        
    def CloseSortFile(self):
        self.DeleteSort()
        self.updateTitle()
        self.EnableSortWidgets(False)
        
    def RestoreClusters2GUI(self):
        """Stuff that needs to be done to synch the GUI with newly imported clusters"""
        self.UpdateClustersGUI() # restore nlist and uslist
        try:
            self.sort.spikes
            # colour points for all clusters in one shot:
            self.ColourPoints(self.sort.clusters.values())
        except AttributeError: pass # no spikes
        self.OpenWindow('Sort')

    def OpenWindow(self, wintype):
        """Create and bind a window, show it, plot its data if applicable. Much of this
        BORDER stuff is just an empirically derived hack"""
        new = wintype not in self.windows
        if new:
            if wintype == 'Spike':
                x = self.pos().x()
                y = self.pos().y() + self.size().height() + WINDOWTITLEHEIGHT
                window = SpikeWindow(parent=self, tw=self.spiketw, pos=(x, y),
                                     size=(self.SPIKEWINDOWWIDTH, SPIKEWINDOWHEIGHT))
            elif wintype == 'Chart':
                x = self.pos().x() + self.SPIKEWINDOWWIDTH + 2*BORDER
                y = self.pos().y() + self.size().height() + WINDOWTITLEHEIGHT
                window = ChartWindow(parent=self, tw=self.charttw, cw=self.spiketw,
                                     pos=(x, y), size=CHARTWINDOWSIZE)
            elif wintype == 'LFP':
                x = self.pos().x() + self.SPIKEWINDOWWIDTH + CHARTWINDOWSIZE[0] + 4*BORDER
                y = self.pos().y() + self.size().height() + WINDOWTITLEHEIGHT
                window = LFPWindow(parent=self, tw=self.lfptw, cw=self.charttw,
                                   pos=(x, y), size=LFPWINDOWSIZE)
            elif wintype == 'Sort':
                x = self.pos().x() + self.size().width() + 2*BORDER
                y = self.pos().y()
                #print('sort x: %d' % x)
                window = SortWindow(parent=self, pos=(x, y))
            elif wintype == 'Cluster':
                x = (self.pos().x() + self.size().width()
                     + self.windows['Sort'].size().width() + 4*BORDER)
                y = self.pos().y()
                size = (SCREENWIDTH - x - 2*BORDER, CLUSTERWINDOWHEIGHT)
                #print('cluster x: %d' % x)
                #print('cluster size: %r' % (size,))
                window = ClusterWindow(parent=self, pos=(x, y), size=size)
            elif wintype == 'MPL':
                x = self.pos().x()
                y = self.pos().y() + self.size().height() + WINDOWTITLEHEIGHT
                window = MPLWindow(parent=self, pos=(x, y),
                                   size=(self.size().width(), self.size().width()))
            self.windows[wintype] = window
            try: # try and load saved window geometry and state from sort
                window.restoreGeometry(self.sort.windowGeometries[wintype])
                window.restoreState(self.sort.windowStates[wintype])
            except (AttributeError, KeyError):
                pass
        self.ShowWindow(wintype) # just show it
        if new: # do stuff that only works after first show
            if wintype not in ['Cluster', 'MPL']:
                window.panel.draw_refs() # prevent plot artifacts
            # should be unnecessary after restoring window state above, but vsplitter
            # and hsplitter aren't restored properly, set them manually:
            if wintype == 'Sort':
                window.mainsplitter.moveSplitter(window.MAINSPLITTERPOS, 1)
                window.vsplitter.moveSplitter(window.VSPLITTERPOS, 1)
        return self.windows[wintype] # 'window' isn't necessarily in local namespace

    def ShowWindow(self, wintype, enable=True):
        """Show/hide a window, force menu and toolbar states to correspond"""
        window = self.windows[wintype]
        if enable:
            window.show()
        else:
            window.hide()
        self.ui.__dict__['action%sWindow' % wintype].setChecked(enable)
        if enable and isinstance(window, DataWindow):
            # update the newly shown data window's data, in case self.t changed since
            # it was last visible
            self.plot(wintype)

    def HideWindow(self, wintype):
        self.ShowWindow(wintype, False)

    def ToggleWindow(self, wintype):
        """Toggle visibility of a data window"""
        try:
            window = self.windows[wintype]
            self.ShowWindow(wintype, not window.isVisible()) # toggle it
        except KeyError: # window hasn't been opened yet
            self.OpenWindow(wintype)

    def CloseWindow(self, wintype):
        """Hide window, remove it from windows dict, destroy it"""
        self.HideWindow(wintype)
        window = self.windows.pop(wintype)
        window.destroy()

    def ToggleRasters(self):
        """Toggle visibility of rasters"""
        enable = self.ui.actionRasters.isChecked()
        self.ShowRasters(enable)

    def ShowRasters(self, enable=True):
        """Show/hide rasters for all applicable windows. Force menu states to correspond"""
        self.ui.actionRasters.setChecked(enable)
        for wintype, window in self.windows.items():
            if wintype in ['Spike', 'Chart', 'LFP']:
                window.panel.show_rasters(enable=enable)
                self.plot(wintype)

    def ToggleRef(self, ref):
        """Toggle visibility of TimeRef, VoltageRef, Scale, or the Caret"""
        enable = self.ui.__dict__['action%s' % ref].isChecked()
        self.ShowRef(ref, enable)

    def ShowRef(self, ref, enable=True):
        """Show/hide a TimeRef, VoltageRef, Scale, or the Caret. Force menu states to
        correspond"""
        self.ui.__dict__['action%s' % ref].setChecked(enable)
        for wintype, window in self.windows.items():
            if wintype in ['Spike', 'Chart', 'LFP', 'Sort']:
                window.panel.show_ref(ref, enable=enable)

    def SetFiltmeth(self, filtmeth):
        """Set highpass filter method"""
        if self.hpstream != None:
            self.hpstream.filtmeth = filtmeth
            self.plot()
        self.ui.__dict__['actionFiltmeth%s' % filtmeth].setChecked(True)

    def SetCAR(self, car):
        """Set common average reference method"""
        if self.hpstream != None:
            self.hpstream.car = car
            self.plot()
        self.ui.__dict__['actionCAR%s' % car].setChecked(True)

    def SetSampfreq(self, sampfreq):
        """Set highpass stream sampling frequency, update widgets"""
        if self.hpstream != None:
            self.hpstream.sampfreq = sampfreq
            self.update_slider() # update slider to account for new tres
            self.plot()
        self.ui.__dict__['action%dkHz' % (sampfreq / 1000)].setChecked(True)

    def SetSHCorrect(self, enable):
        """Set highpass stream sample & hold correct flag, update widgets"""
        if self.hpstream != None:
            self.hpstream.shcorrect = enable
        self.ui.actionSampleAndHoldCorrect.setChecked(enable)
        self.plot()

    def EnableStreamWidgets(self, enable):
        """Enable/disable all widgets that require an open stream"""
        try:
            self.sort
        except AttributeError:
            # change these menu states only if sort doesn't already exist:
            self.EnableFilteringMenu(enable)
            self.EnableCARMenu(enable)
            self.EnableSamplingMenu(enable)
        self.EnableConvertMenu(enable)
        self.ui.filePosStartButton.setEnabled(enable)
        self.ui.filePosLineEdit.setEnabled(enable)
        self.ui.filePosEndButton.setEnabled(enable)
        self.ui.slider.setEnabled(enable)
        self.ui.detectButton.setEnabled(enable)

    def EnableSortWidgets(self, enable):
        """Enable/disable all widgets that require a sort"""
        self.EnableFilteringMenu(not enable)
        self.EnableCARMenu(not enable)
        self.EnableSamplingMenu(not enable)
        self.ui.actionRasters.setEnabled(enable)
        self.ShowRasters(enable)
        self.ui.tabWidget.setCurrentIndex(int(enable)) # select cluster or detect tab
        self.EnableSpikeWidgets(enable)

    def EnableFilteringMenu(self, enable):
        """Enable/disable all items in Filtering menu, while still allowing
        the menu to be opened and its contents viewed"""
        for action in self.ui.menuFiltering.actions():
            action.setEnabled(enable)

    def EnableCARMenu(self, enable):
        """Enable/disable all items in CAR menu, while still allowing
        the menu to be opened and its contents viewed"""
        for action in self.ui.menuCAR.actions():
            action.setEnabled(enable)

    def EnableSamplingMenu(self, enable):
        """Enable/disable all items in Sampling menu, while still allowing
        the menu to be opened and its contents viewed"""
        for action in self.ui.menuSampling.actions():
            action.setEnabled(enable)

    def EnableConvertMenu(self, enable):
        """Enable/disable all items in Convert menu, while still allowing
        the menu to be opened and its contents viewed"""
        for action in self.ui.menuConvert.actions():
            action.setEnabled(enable)

    def EnableSpikeWidgets(self, enable):
        """Enable/disable all widgets that require the current Sort to have spikes"""
        return # do nothing for now
        '''
        try:
            if len(self.sort.spikes) == 0: enable = False # no spikes
        except AttributeError: enable = False # self.sort doesn't exist yet
        self.extract_pane.Enable(enable)
        try: self.sort.extractor
        except AttributeError: enable = False # no params extracted, or .sort doesn't exist
        self.cluster_pane.Enable(enable)
        try:
            if len(self.sort.clusters) == 0: enable = False # no clusters exist yet
        except AttributeError: enable = False
        self.cluster_params_pane.Enable(enable)
        try:
            if len(self.sort.neurons) == 0: enable = False # no neurons
        except AttributeError: enable = False # self.sort doesn't exist yet
        self.validate_pane.Enable(enable)
        '''
    def get_detector(self):
        """Create and bind Detector object, update sort from gui"""
        self.sort.detector = Detector(sort=self.sort)
        self.update_sort_from_gui()

    def update_dirtysids(self, sids):
        """Update self.dirtysids and clear the dimension reduction cache"""
        self.dirtysids.update(sids)
        # clear the dimension reduction cache:
        self.sort.X = {}

    def update_spiketw(self, spiketw):
        """Update tw of self.sort and of Spike and Sort windows. For efficiency,
        only update sort and windows when necessary. This is appropriate
        for the user to call directly from the command line."""
        assert len(spiketw) == 2
        assert spiketw[0] < 0 and spiketw[1] > 0
        self.spiketw = spiketw
        if hasattr(self, 'sort'):
            if self.sort.tw != spiketw:
                self.sort.update_tw(spiketw)
        for wintype in ['Spike', 'Sort']:
            if wintype in self.windows:
                panel = self.windows[wintype].panel
                if panel.tw != spiketw:
                    panel.update_tw(spiketw)
 
    def update_sort_from_gui(self):
        self.update_sort_from_detector_pane()
        self.update_sort_from_cluster_pane()

    def update_sort_from_detector_pane(self):
        ui = self.ui
        det = self.sort.detector
        det.chans = self.chans_enabled
        if ui.globalFixedRadioButton.isChecked():
            threshmethod = 'GlobalFixed'
        elif ui.channelFixedRadioButton.isChecked():
            threshmethod = 'ChanFixed'
        elif ui.dynamicRadioButton.isChecked():
            threshmethod = 'Dynamic'
        else:
            raise ValueError
        det.threshmethod = threshmethod
        det.fixedthreshuV = ui.globalFixedSpinBox.value()
        det.noisemult = ui.dynamicNoiseXSpinBox.value()
        det.noisemethod = str(ui.noiseMethodComboBox.currentText())
        det.ppthreshmult = ui.vppThreshXSpinBox.value()
        det.dt = ui.dtSpinBox.value()
        det.trange = self.get_detectortrange()
        det.blocksize = int(float(ui.blockSizeLineEdit.text())) # allow exp notation
        det.lockrx = ui.lockRxSpinBox.value()
        det.inclr = ui.inclRSpinBox.value()

    def update_sort_from_cluster_pane(self):
        ui = self.ui
        s = self.sort
        s.sigma = ui.sigmaSpinBox.value()
        s.rmergex = ui.rmergeXSpinBox.value()
        s.rneighx = ui.rneighXSpinBox.value()
        s.alpha = ui.alphaSpinBox.value()
        s.maxgrad = ui.maxgradSpinBox.value()
        s.minpoints = ui.minpointsSpinBox.value()

    def update_gui_from_sort(self):
        ui = self.ui
        s = self.sort
        det = s.detector
        if self.hpstream:
            self.chans_enabled = det.chans
        # update detector pane
        meth2widget = {'GlobalFixed': ui.globalFixedRadioButton,
                       'ChanFixed': ui.channelFixedRadioButton,
                       'Dynamic': ui.dynamicRadioButton}
        meth2widget[det.threshmethod].setChecked(True)
        ui.globalFixedSpinBox.setValue(det.fixedthreshuV)
        ui.dynamicNoiseXSpinBox.setValue(det.noisemult)
        ui.noiseMethodComboBox.setCurrentIndex(ui.noiseMethodComboBox.findText(det.noisemethod))
        ui.vppThreshXSpinBox.setValue(det.ppthreshmult)
        ui.dtSpinBox.setValue(det.dt)
        ui.rangeStartLineEdit.setText(str(det.trange[0]))
        ui.rangeEndLineEdit.setText(str(det.trange[1]))
        ui.blockSizeLineEdit.setText(str(det.blocksize))
        ui.lockRxSpinBox.setValue(det.lockrx)
        ui.inclRSpinBox.setValue(det.inclr)
        # update cluster pane
        ui.sigmaSpinBox.setValue(s.sigma)
        ui.rmergeXSpinBox.setValue(s.rmergex)
        ui.rneighXSpinBox.setValue(s.rneighx)
        ui.alphaSpinBox.setValue(s.alpha)
        ui.maxgradSpinBox.setValue(s.maxgrad)
        ui.minpointsSpinBox.setValue(s.minpoints)

    def get_detectortrange(self):
        """Get detector time range from combo boxes, and convert
        start, now, and end to appropriate vals"""
        t0 = str(self.ui.rangeStartLineEdit.text())
        t1 = str(self.ui.rangeEndLineEdit.text())
        try:
            t0 = self.str2t[t0]
        except KeyError:
            t0 = int(float(t0)) # convert to float to allow exp notation shorthand
        try:
            t1 = self.str2t[t1]
        except KeyError:
            t1 = int(float(t1))
        return t0, t1

    def get_nearest_timepoint(self, t):
        """Round t to nearest (possibly interpolated) sample timepoint"""
        t = intround(t / self.hpstream.tres) * self.hpstream.tres
        t = min(max(t, self.trange[0]), self.trange[1]) # constrain to within self.trange
        return t

    def seek(self, t=0):
        """Seek to position in stream. t is time in us"""
        # for some reason, sometimes seek is called during spyke's shutdown process,
        # after hpstream has been removed. This prevents raising an error:
        if self.hpstream == None:
            return
        oldt = self.t
        self.t = self.get_nearest_timepoint(t)
        self.str2t['now'] = self.t # update
        # only plot if t has actually changed, though this doesn't seem to improve
        # performance, maybe mpl is already doing something like this?
        if self.t != oldt: # update controls first so they don't lag
            self.ui.filePosLineEdit.setText('%.1f' % self.t)
            self.ui.slider.setValue(intround(self.t / self.hpstream.tres))
            self.plot()
    
    def step(self, direction):
        """Step one timepoint left or right"""
        self.seek(self.t + direction*self.hpstream.tres)

    def tell(self):
        """Return current position in data file"""
        return self.t

    def plot(self, wintypes=None):
        """Update the contents of all the data windows, or just specific ones.
        Center each data window on self.t"""
        if wintypes == None: # update all visible windows
            wintypes = self.windows.keys()
        else: # update only specific windows, if visible
            wintypes = toiter(wintypes)
        # reorder:
        wintypes = [ wintype for wintype in WINDOWUPDATEORDER if wintype in wintypes ]
        windows = [ self.windows[wintype] for wintype in wintypes ] # get windows in order
        for wintype, window in zip(wintypes, windows):
            if window.isVisible(): # for performance, only update if window is shown
                if wintype == 'Spike':
                    wave = self.hpstream(self.t+self.spiketw[0], self.t+self.spiketw[1])
                elif wintype == 'Chart':
                    wave = self.hpstream(self.t+self.charttw[0], self.t+self.charttw[1])
                elif wintype == 'LFP':
                    wave = self.lpstream(self.t+self.lfptw[0], self.t+self.lfptw[1])
                window.panel.plot(wave, tref=self.t) # plot it


class DataWindow(SpykeToolWindow):
    """Base data window to hold a custom spyke panel widget"""
    def setupUi(self, pos, size):
        self.setCentralWidget(self.panel)
        self.resize(*size)
        self.move(*pos)

    def step(self, direction):
        """Step left or right one caret width"""
        panelwidth = self.panel.cw[1] - self.panel.cw[0]
        spw = self.parent()
        spw.seek(spw.t + direction * panelwidth)

    def page(self, direction):
        """Page left or right one panel width"""
        panelwidth = self.panel.tw[1] - self.panel.tw[0]
        spw = self.parent()
        spw.seek(spw.t + direction * panelwidth)

    def keyPressEvent(self, event):
        spw = self.parent()
        key = event.key()
        if key == Qt.Key_Left:
            self.step(-1)
        elif key == Qt.Key_Right:
            self.step(+1)
        elif key == Qt.Key_PageUp:
            self.page(-1)
        elif key == Qt.Key_PageDown:
            self.page(+1)
        else:
            SpykeToolWindow.keyPressEvent(self, event) # pass it on


class SpikeWindow(DataWindow):
    """Window to hold the custom spike panel widget"""
    def __init__(self, parent=None, tw=None, cw=None, pos=None, size=None):
        DataWindow.__init__(self, parent)
        self.panel = SpikePanel(self, tw=tw, cw=cw)
        self.setupUi(pos, size)
        self.setWindowTitle("Spike Window")

    def step(self, direction):
        """Step left or right one sample timepoint"""
        spw = self.parent()
        spw.step(direction)

    def keyPressEvent(self, event):
        spw = self.parent()
        key = event.key()
        ctrl = event.modifiers() == Qt.ControlModifier # only modifier is ctrl
        if ctrl and key in [Qt.Key_Enter, Qt.Key_Return]:
            self.panel.reloadSelectedSpike()
        else:
            DataWindow.keyPressEvent(self, event) # pass it on


class ChartWindow(DataWindow):
    """Window to hold the custom chart panel widget"""
    def __init__(self, parent=None, tw=None, cw=None, pos=None, size=None):
        DataWindow.__init__(self, parent)
        self.panel = ChartPanel(self, tw=tw, cw=cw)
        self.setupUi(pos, size)
        self.setWindowTitle("Chart Window")


class LFPWindow(DataWindow):
    """Window to hold the custom LFP panel widget"""
    def __init__(self, parent=None, tw=None, cw=None, pos=None, size=None):
        DataWindow.__init__(self, parent)
        self.panel = LFPPanel(self, tw=tw, cw=cw)
        self.setupUi(pos, size)
        self.setWindowTitle("LFP Window")


class MPLWindow(SpykeToolWindow):
    """Matplotlib window"""
    def __init__(self, parent=None, pos=None, size=None):
        SpykeToolWindow.__init__(self, parent)
        figure = Figure()
        self.f = figure
        self.figurecanvas = FigureCanvas(figure)
        self.setCentralWidget(self.figurecanvas)
        self.toolbar = NavigationToolbar(self.figurecanvas, self, False)
        self.toolbar.setObjectName('toolbar')
        self.addToolBar(self.toolbar)
        QtCore.QObject.connect(self.toolbar, QtCore.SIGNAL("message"),
                               self.statusBar().showMessage)
        self.resize(*size)
        self.move(*pos)
        self.setWindowTitle("MPL Window")
        self.ax = figure.add_subplot(111)

class Match(object):
    """Just an object to store rmserror calculations between all clusters
    and all unsorted spikes, and also to store which cluster each spike
    matches best"""
    def __init__(self, cids=None, sids=None, errs=None):
        self.cids = cids # row labels
        self.sids = sids # column labels
        self.errs = errs # len(cids) x len(sids) error array
        self.best = {} # dict with cluster ids as keys and sids as values
        bestcidis = errs.argmin(axis=0) # of length len(sids)
        for cidi, cid in enumerate(cids):
            sidis, = np.where(bestcidis == cidi)
            self.best[cid] = sids[sidis]

    def get_best_errs(self, cid):
        """Get rmserror values between cluster cid and all the unsorted spikes
        in self.sids that match it best"""
        cidi = self.cids.searchsorted(cid)
        bestsids = self.best[cid]
        bestsidis = self.sids.searchsorted(bestsids)
        return self.errs[cidi, bestsidis]
        

if __name__ == '__main__':
    # prevents "The event loop is already running" messages when calling ipshell():
    QtCore.pyqtRemoveInputHook()
    app = QtGui.QApplication(sys.argv)
    spykewindow = SpykeWindow()
    spykewindow.show()
    sys.exit(app.exec_())
