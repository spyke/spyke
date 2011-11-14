"""Define the Cluster object and cluster window"""

from __future__ import division

__authors__ = ['Martin Spacek']

import sys
import time
import random
import numpy as np

from PyQt4 import QtCore, QtGui, QtOpenGL, uic
from PyQt4.QtCore import Qt
from OpenGL import GL, GLU

from core import SpykeToolWindow, lstrip, lst2shrtstr, tocontig
from plot import CMAP, GREYRGB

CLUSTERPARAMSAMPLESIZE = 1000
VIEWDISTANCE = 50


class Cluster(object):
    """Just a simple container for scaled multidim cluster parameters. A
    Cluster will always correspond to a Neuron"""
    def __init__(self, neuron):
        self.neuron = neuron
        self.pos = {'x0':0, 'y0':0, 'sx':0, 'sy':0, 'Vpp':0, 'V0':0, 'V1':0,
                    'dphase':0, 't':0, 's0':0, 's1':0}
        # cluster normpos are scaled values, suitable for plotting
        self.normpos = {'x0':0, 'y0':0, 'sx':0, 'sy':0, 'Vpp':0, 'V0':0, 'V1':0,
                        'dphase':0, 't':0, 's0':0, 's1':0}

    def get_id(self):
        return self.neuron.id

    def set_id(self, id):
        self.neuron.id = id

    id = property(get_id, set_id)

    def get_color(self):
        if self.id < 1:
            return GREYRGB # unclustered or multiunit
        else:
            return CMAP[self.id % len(CMAP) - 1] # single unit nids are 1-based

    color = property(get_color)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        # don't save any temporary principal component positions
        pos = self.pos.copy() # don't modify original
        normpos = self.normpos.copy() # don't modify original
        assert sorted(pos) == sorted(normpos) # make sure they have same set of keys
        for key in list(pos): # need list() for snapshot of keys before any are deleted
            if key.startswith('c') and key[-1].isdigit():
                del pos[key]
                del normpos[key]
        d['pos'] = pos
        d['normpos'] = normpos
        return d

    def update_pos(self, dims=None, nsamples=CLUSTERPARAMSAMPLESIZE):
        """Update unnormalized and normalized cluster position for specified dims.
        Use median instead of mean to reduce influence of outliers on cluster
        position. Subsample for speed"""
        sort = self.neuron.sort
        spikes = sort.spikes
        if dims == None: # use all of them
            dims = list(self.pos) # some of these might not exist in spikes array
        sids = self.neuron.sids
        if nsamples and len(sids) > nsamples: # subsample spikes
            print('neuron %d: update_pos() taking random sample of %d spikes instead '
                  'of all %d of them' % (self.id, nsamples, len(sids)))
            sids = np.asarray(random.sample(sids, nsamples))

        # check for pre-calculated spike param means and stds
        try: sort.means
        except AttributeError: sort.means = {}
        try: sort.stds
        except AttributeError: sort.stds = {}

        ## FIXME: some code duplication from sort.get_param_matrix()?
        for dim in dims:
            try:
                spikes[dim]
            except ValueError:
                continue # this dim doesn't exist in spikes record array, ignore it
            # data from all spikes:
            data = spikes[dim]
            # data from neuron's spikes, potentially subsample of them,
            # copied for in-place normalization:
            subdata = data[sids].copy()
            # update unnormalized position
            self.pos[dim] = np.median(subdata)
            # calculate mean and std for normalization
            try: mean = sort.means[dim]
            except KeyError:
                mean = data.mean()
                sort.means[dim] = mean # save to pre-calc
            if dim in ['x0', 'y0']: # normalize spatial params by x0 std
                try: std = sort.stds['x0']
                except KeyError:
                    std = spikes['x0'].std()
                    sort.stds['x0'] = std # save to pre-calc
            else: # normalize all other params by their std
                try: std = sort.stds[dim]
                except KeyError:
                    std = data.std()
                    sort.stds[dim] = std # save to pre-calc
            # now do the actual normalization
            subdata -= mean
            if std != 0:
                subdata /= std
            # update normalized position
            self.normpos[dim] = np.median(subdata)

    def update_comppos(self, nsamples=CLUSTERPARAMSAMPLESIZE):
        """Update component analysis (PCA/ICA) values for self"""
        sort = self.neuron.sort
        #comp = sort.get_component_matrix()
        comp = sort.comp
        ncomp = comp.shape[1]
        compsids = sort.compsids
        nsids = self.neuron.sids
        # consider only nsids that were included in last CA:
        nsids = np.intersect1d(nsids, compsids, assume_unique=True)
        if nsamples and len(nsids) > nsamples: # subsample spikes
            print('neuron %d: update_comppos() taking random sample of %d spikes instead '
                  'of all %d that were included in last CA' % (self.id, nsamples, len(nsids)))
            nsids = np.asarray(random.sample(nsids, nsamples))
        compsidis = compsids.searchsorted(nsids)
        subcomp = comp[compsidis].copy()
        medians = np.median(subcomp, axis=0)
        mean = comp.mean(axis=0)
        std = comp.std(axis=0)
        subcomp -= mean
        subcomp /= std
        normmedians = np.median(subcomp, axis=0)
        # write comp fields to dicts:
        for compid in range(ncomp):
            dim = 'c%d' % compid
            self.pos[dim] = medians[compid]
            self.normpos[dim] = normmedians[compid]

class ClusterWindow(SpykeToolWindow):
    def __init__(self, parent, pos=None, size=None):
        SpykeToolWindow.__init__(self, parent, flags=QtCore.Qt.Tool)
        self.spykewindow = parent
        self.setWindowTitle("Cluster Window")
        self.move(*pos)
        self.resize(*size)

        self.glWidget = GLWidget(parent=self)
        self.setCentralWidget(self.glWidget)

    def closeEvent(self, event):
        self.spykewindow.HideWindow('Cluster')

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F11:
            SpykeToolWindow.keyPressEvent(self, event) # pass it up
        else:
            self.glWidget.keyPressEvent(event) # pass it down

    def plot(self, X, sids, nids):
        """Plot 3D projection of (possibly clustered) spike params in X"""
        X = tocontig(X) # ensure it's contig
        gw = self.glWidget
        gw.points = X
        gw.npoints = len(X)
        gw.sids = sids
        gw.nids = nids
        gw.colors = CMAP[nids % len(CMAP) - 1] # uint8, single unit nids are 1-based
        gw.colors[nids < 1] = GREYRGB # overwrite unclustered/multiunit points with GREYRGB
        gw.updateGL()


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.spw = self.parent().spykewindow
        #self.setMouseTracking(True) # req'd for tooltips purely on mouse motion, slow
        self.lastPos = QtCore.QPoint()
        self.focus = np.float32([0, 0, 0]) # init camera focus

        format = QtOpenGL.QGLFormat()
        format.setDoubleBuffer(True) # req'd for picking
        self.setFormat(format)

    def get_sids(self):
        return self._sids

    def set_sids(self, sids):
        """Set up rgbsids array for later use in self.pick()"""
        self._sids = sids
        # encode sids in RGB
        r = sids // 256**2
        rem = sids % 256**2 # remainder
        g = rem // 256
        b = rem % 256
        self.rgbsids = np.zeros((self.npoints, 3), dtype=np.uint8)
        self.rgbsids[:, 0] = r
        self.rgbsids[:, 1] = g
        self.rgbsids[:, 2] = b

    sids = property(get_sids, set_sids)

    def initializeGL(self):
        # these are the defaults anyway, but just to be thorough:
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClearDepth(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST) # display points according to occlusion, not order of plotting
        '''
        #GL.glEnable(GL.GL_POINT_SMOOTH) # doesn't seem to work right, proper way to antialiase?
        #GL.glEnable(GL.GL_LINE_SMOOTH) # works better
        #GL.glPointSize(1.5) # truncs to the nearest pixel if antialiasing is off
        '''
        # set initial position and orientation of camera
        GL.glTranslate(0, 0, -VIEWDISTANCE)
        GL.glRotate(-45, 0, 0, 1)
        GL.glRotate(-45, 0, 1, 0)

    def reset(self):
        """Stop plotting"""
        self.npoints = 0
        self.updateGL()

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Don't load identity matrix. Do all transforms in place against current matrix
        # and take advantage of OpenGL's state-machineness.
        #GL.glLoadIdentity() # loads identity matrix into top of matrix stack

        GL.glEnableClientState(GL.GL_COLOR_ARRAY);
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
        GL.glColorPointerub(self.colors) # should be n x rgb uint8, ie usigned byte
        GL.glVertexPointerf(self.points) # should be n x 3 contig float32
        GL.glDrawArrays(GL.GL_POINTS, 0, self.npoints)
        # doesn't seem to be necessary, even though I'm in double-buffered mode with the
        # back buffer for RGB sid encoding, but do it anyway for completeness
        self.swapBuffers()

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        # fov (deg) controls amount of perspective, and as a side effect initial apparent size
        GLU.gluPerspective(45, width/height, 0.0001, 1000) # fov, aspect, nearz & farz clip planes
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def get_MV(self):
        """Return modelview matrix"""
        return GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)

    def set_MV(self, MV):
        GL.glLoadMatrixd(MV)

    MV = property(get_MV, set_MV)

    # modelview matrix is column major, so we work on columns instead of rows
    def getViewRight(self):
        """View right vector: 1st col of modelview matrix"""
        return self.MV[:3, 0]

    def getViewUp(self):
        """View up vector: 2nd col of modelview matrix"""
        return self.MV[:3, 1]

    def getViewNormal(self):
        """View normal vector: 3rd col of modelview matrix"""
        return self.MV[:3, 2]

    def getTranslation(self):
        """Translation vector: 4th row of modelview matrix"""
        return self.MV[3, :3]

    def setTranslation(self, vt):
        """Translation vector: 4th row of modelview matrix"""
        MV = self.MV
        MV[3, :3] = vt
        self.MV = MV

    def getDistance(self):
        v = self.getTranslation()
        # for pan and zoom, doesn't seem to matter whether d is from origin or focus
        #return np.sqrt((v**2).sum()) # from data origin
        return np.sqrt(((v-self.focus)**2).sum()) # from focus

    def pan(self, dx, dy):
        """Translate along view right and view up vectors"""
        d = self.getDistance()
        vr = self.getViewRight()
        vr *= dx*d
        GL.glTranslate(vr[0], vr[1], vr[2])
        vu = self.getViewUp()
        vu *= dy*d
        GL.glTranslate(vu[0], vu[1], vu[2])

    def zoom(self, dr):
        """Translate along view normal vector"""
        d = self.getDistance()
        vn = self.getViewNormal()
        vn *= dr*d
        GL.glTranslate(vn[0], vn[1], vn[2])

    def pitch(self, dangle): # aka elevation
        """Rotate around view right vector"""
        vr = self.getViewRight()
        GL.glTranslate(*self.focus)
        GL.glRotate(dangle, *vr)
        GL.glTranslate(*-self.focus)

    def yaw(self, dangle): # aka azimuth
        """Rotate around view up vector"""
        vu = self.getViewUp()
        GL.glTranslate(*self.focus)
        GL.glRotate(dangle, *vu)
        GL.glTranslate(*-self.focus)

    def roll(self, dangle):
        """Rotate around view normal vector"""
        vn = self.getViewNormal()
        GL.glTranslate(*self.focus)
        GL.glRotate(dangle, *vn)
        GL.glTranslate(*-self.focus)

    def panTo(self, p=None):
        """Translate along view right and view up vectors such that data point p is
        centered in the viewport. Not entirely sure why or how this works, figured
        it out using guess and test"""
        if p == None:
            p = self.focus
        MV = self.MV
        vr = self.getViewRight()
        vu = self.getViewUp()
        p = -p
        x = np.dot(p, vr) # dot product
        y = np.dot(p, vu)
        MV[3, :2] = x, y # set first two entries of 4th row to x, y
        self.MV = MV

    def pick(self, x, y):
        """Return sid of point at window coords x, y (bottom left origin)"""
        width = self.size().width()
        height = self.size().height()
        #print('coords: %d, %d' % (x, y))
        # constrain to within border 1 pix smaller than widget, for glReadPixels call
        if not (1 <= x < width-1 and 1 <= y < height-1): # cursor out of range
            return
        if self.npoints > 2**24-2: # the last one is the full white background used as a no hit
            raise OverflowError("Can't pick from more than 2**24-2 sids")
        # draw encoded RGB values to back buffer
        #GL.glDrawBuffer(GL_BACK) # defaults to back
        GL.glClearColor(1.0, 1.0, 1.0, 1.0) # highest possible RGB means no hit
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY);
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
        GL.glColorPointerub(self.rgbsids) # unsigned byte, ie uint8
        GL.glVertexPointerf(self.points) # float32
        GL.glDrawArrays(GL.GL_POINTS, 0, self.npoints) # to back buffer
        GL.glClearColor(0.0, 0.0, 0.0, 1.0) # restore to default black
        # grab back buffer
        #GL.glReadBuffer(GL.GL_BACK) # defaults to back
        # find rgb at or around cursor coords, decode sid
        backbuffer = GL.glReadPixelsub(x-1, y-1, 3, 3, GL.GL_RGB) # unsigned byte
        if (backbuffer == 255).all(): # no hit
            return
        sid = self.decodeRGB(backbuffer[1, 1])
        if sid != None:
            #print('hit at exact cursor pos')
            return sid # hit at exact cursor position
        hitpix = (backbuffer != [255, 255, 255]).sum(axis=2) # 2D array with nonzero entries at hits
        ri = np.where(hitpix.ravel())[0][0] # get ravelled index of first hit
        i, j = np.unravel_index(ri, dims=hitpix.shape) # unravel to 2D index
        #print('hit at %d, %d' % (i, j))
        return self.decodeRGB(backbuffer[i, j]) # should be a valid sid

    def decodeRGB(self, rgb):
        """Convert encoded rgb value to sid"""
        r, g, b = rgb
        sid = r*65536 + g*256 + b
        if sid != 16777215: # 2**24 - 1
            return sid # it's a valid sid

    def cursorPosQt(self):
        """Get current mouse cursor position in Qt coords (top left origin)"""
        globalPos = QtGui.QCursor.pos()
        pos = self.mapFromGlobal(globalPos)
        return pos.x(), pos.y()

    def cursorPosGL(self):
        """Get current mouse cursor position in OpenGL coords (bottom left origin)"""
        globalPos = QtGui.QCursor.pos()
        pos = self.mapFromGlobal(globalPos)
        y = self.size().height() - pos.y()
        return pos.x(), y

    def GLtoQt(self, x, y):
        """Convert GL screen coords to Qt, return as QPoint"""
        y = self.size().height() - y
        return QtCore.QPoint(x, y)

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseDoubleClickEvent(self, event):
        """Clear selection and select spike and/or cluster under the cursor, if any"""
        self.selectItemUnderCursor(clear=True)

    def mouseMoveEvent(self, event):
        buttons = event.buttons()
        modifiers = event.modifiers()
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if buttons == QtCore.Qt.LeftButton:
            if modifiers == Qt.ControlModifier:
                self.roll(-0.5*dx - 0.5*dy)
            elif modifiers == Qt.ShiftModifier:
                self.pan(dx/700, -dy/700) # qt viewport y axis points down
            else:
                self.yaw(0.5*dx)
                self.pitch(0.5*dy)
        elif buttons == QtCore.Qt.RightButton:
            self.zoom(-dy/500) # qt viewport y axis points down

        # pop up a tooltip on mouse movement, requires mouse tracking enabled
        if buttons == Qt.NoButton:
            self.showToolTip()
        else:
            QtGui.QToolTip.hideText()

        self.updateGL()
        self.lastPos = QtCore.QPoint(event.pos())

    def wheelEvent(self, event):
        self.zoom(event.delta() / 2000)
        self.updateGL()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        sw = self.spw.windows['Sort']
        ctrldown = bool(Qt.ControlModifier & modifiers)
        shiftdown = bool(Qt.ShiftModifier & modifiers)
        
        if ctrldown and key not in [Qt.Key_Enter, Qt.Key_Return]:
            if key == Qt.Key_Left:
                self.roll(5)
            elif key == Qt.Key_Right:
                self.roll(-5)
            elif key == Qt.Key_Up:
                self.zoom(0.05)
            elif key == Qt.Key_Down:
                self.zoom(-0.05)
        elif shiftdown and key != Qt.Key_NumberSign:
            if key == Qt.Key_Left:
                self.pan(-0.05, 0)
            elif key == Qt.Key_Right:
                self.pan(0.05, 0)
            elif key == Qt.Key_Up:
                self.pan(0, 0.05)
            elif key == Qt.Key_Down:
                self.pan(0, -0.05)
        else:
            if key == Qt.Key_Left:
                self.yaw(-5)
            elif key == Qt.Key_Right:
                self.yaw(5)
            elif key == Qt.Key_Up:
                self.pitch(-5)
            elif key == Qt.Key_Down:
                self.pitch(5)
            elif key in (Qt.Key_P, Qt.Key_T): # 'pick' or 'tooltip'
                #print(self.pick(*self.cursorPosGL()))
                self.showToolTip()
            elif key == Qt.Key_0: # reset focus to origin
                self.focus = np.float32([0, 0, 0])
                self.panTo() # pan to new focus
            elif key == Qt.Key_F: # reset focus to cursor position
                sid = self.pick(*self.cursorPosGL())
                if sid != None:
                    self.focus = self.points[self.sids.searchsorted(sid)]
                    self.panTo() # pan to new focus
            elif key == Qt.Key_S: # select item under the cursor, if any
                self.selectItemUnderCursor(on=True, clear=False)
            elif key == Qt.Key_D: # deselect item under the cursor, if any
                self.selectItemUnderCursor(on=False, clear=False)
            #elif key == Qt.Key_Space: # clear and select item under cursor, if any
            #    self.selectItemUnderCursor(on=True, clear=True)
            elif key in [Qt.Key_Escape, Qt.Key_Delete, Qt.Key_M, Qt.Key_Slash, Qt.Key_Backslash,
                         Qt.Key_NumberSign, Qt.Key_C, Qt.Key_X, Qt.Key_R, Qt.Key_Space,
                         Qt.Key_B, Qt.Key_Comma, Qt.Key_Period, Qt.Key_H]:
                sw.keyPressEvent(event) # pass it on to Sort window
            elif key in [Qt.Key_Enter, Qt.Key_Return]:
                sw.spykewindow.ui.plotButton.click() # same as hitting ENTER in nslist
            elif key == Qt.Key_F11:
                self.parent().keyPressEvent(event) # pass it on to parent Cluster window
            elif key == Qt.Key_Backslash:
                self.showProjectionDialog()            

        self.updateGL()

    def showToolTip(self):
        """Pop up a nid or sid tooltip at current mouse cursor position"""
        #QtGui.QToolTip.hideText() # hide first if you want tooltip to move even when text is unchanged
        spw = self.spw
        sort = spw.sort
        x, y = self.cursorPosGL()
        sid = self.pick(x, y)
        if sid != None:
            spos = []
            dims = spw.GetClusterPlotDims()
            for dim in dims:
                if dim.startswith('c') and dim[-1].isdigit():
                    compid = int(lstrip(dim, 'c'))
                    compsidi = sort.compsids.searchsorted(sid)
                    spos.append(sort.comp[compsidi, compid])
                else: # it's a standard dim stored in spikes array
                    spos.append(sort.spikes[sid][dim])
            tip = 'sid: %d' % sid
            tip += '\n%s: %s' % (lst2shrtstr(dims), lst2shrtstr(spos))
            nid = sort.spikes[sid]['nid']
            if nid != 0:
                tip += '\nnid: %d' % nid
                cpos = [ sort.neurons[nid].cluster.pos[dim] for dim in dims ]
                tip += '\n%s: %s' % (lst2shrtstr(dims), lst2shrtstr(cpos))
            globalPos = self.mapToGlobal(self.GLtoQt(x, y))
            QtGui.QToolTip.showText(globalPos, tip)
        else:
            QtGui.QToolTip.hideText()

    def selectItemUnderCursor(self, on=True, clear=False):
        spw = self.spw
        sw = spw.windows['Sort']
        if clear:
            sw.uslist.clearSelection()
            sw.nlist.clearSelection()
        globalPos = QtGui.QCursor.pos()
        pos = self.mapFromGlobal(globalPos)
        x = pos.x()
        y = self.size().height() - pos.y()
        sid = self.pick(x, y)
        if sid != None:
            spw.SelectSpike(sid, on=on) # select/deselect spike & its cluster too, if need be
        #self.showToolTip()

    def showProjectionDialog(self):
        """Get and set OpenGL ModelView matrix and focus.
        Useful for setting two different instances to the exact same projection"""
        dlg = uic.loadUi('multilineinputdialog.ui')
        dlg.setWindowTitle('Get and set OpenGL ModelView matrix and focus')
        precision = 8 # use default precision
        MV_repr = np.array_repr(self.MV, precision=precision)
        focus_repr = np.array_repr(self.focus, precision=precision)
        txt = ("self.MV = \\\n"
               "%s\n\n"
               "self.focus = %s" % (MV_repr, focus_repr))
        dlg.plainTextEdit.insertPlainText(txt)
        dlg.plainTextEdit.selectAll()
        if dlg.exec_(): # returns 1 if OK, 0 if Cancel
            txt = str(dlg.plainTextEdit.toPlainText())
            from numpy import array, float32 # required for exec()
            exec(txt) # update self.MV and self.focus, with hopefully no maliciousness
