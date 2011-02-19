"""Define the Cluster object and cluster window"""

from __future__ import division

__authors__ = ['Martin Spacek']

#import os
#os.environ['ETS_TOOLKIT'] = 'qt4'
import sys
import time
import random
import numpy as np

from PyQt4 import QtCore, QtGui, QtOpenGL
from PyQt4.QtCore import Qt
from OpenGL import GL, GLU

from core import SpykeToolWindow, lst2shrtstr, normdeg
from plot import CMAP, GREY

CLUSTERPARAMSAMPLESIZE = 1000
VIEWDISTANCE = 50


class Cluster(object):
    """Just a simple container for scaled multidim cluster parameters. A
    Cluster will always correspond to a Neuron"""
    def __init__(self, neuron):
        self.neuron = neuron
        # cluster attribs store scaled values of each dim, suitable for plotting
        self.pos = {'x0':0, 'y0':0, 'sx':0, 'sy':0, 'Vpp':0, 'V0':0, 'V1':0, 'dphase':0, 't':0, 's0':0, 's1':0}
        # for ori, each dict entry for each dim is (otherdim1, otherdim2): ori_value
        # reversing the dims in the key requires negating the ori_value
        self.ori = {'x0':{}, 'y0':{}, 'sx':{}, 'sy':{}, 'Vpp':{}, 'V0':{}, 'V1':{}, 'dphase':{}, 't':{}, 's0':{}, 's1':{}}
        # set scale to 0 to exclude a param from consideration as a
        # dim when checking which points fall within which ellipsoid
        self.scale = {'x0':0.25, 'y0':0.25, 'sx':0, 'sy':0, 'Vpp':0.25, 'V0':0, 'V1':0, 'dphase':0, 't':0, 's0':0, 's1':0}

    def get_id(self):
        return self.neuron.id

    def set_id(self, id):
        self.neuron.id = id

    id = property(get_id, set_id)

    def updatePosScale(self, dims=None, nsamples=CLUSTERPARAMSAMPLESIZE):
        """Update normalized cluster position and scale for specified dims. Use median
        instead of mean to reduce influence of outliers on cluster position.
        Subsample for speed"""
        sort = self.neuron.sort
        spikes = sort.spikes
        if dims == None: # use all of them
            dims = list(self.pos) # some of these might not exist in spikes array
        sids = self.neuron.sids
        if nsamples and len(sids) > nsamples: # subsample spikes
            print('neuron %d: updatePosScale() taking random sample of %d spikes instead '
                  'of all %d of them' % (self.id, nsamples, len(sids)))
            sids = np.asarray(random.sample(sids, nsamples))

        # check for pre-calculated spike param means and stds
        try: sort.means
        except AttributeError: sort.means = {}
        try: sort.stds
        except AttributeError: sort.stds = {}

        for dim in dims:
            try:
                spikes[dim]
            except ValueError:
                continue # this dim doesn't exist in spikes record array, ignore it
            # create normalized copy of all samples for this dim, sort of duplicating code
            # in sort.get_param_matrix()
            data = spikes[dim]
            subdata = data[sids].copy() # subsampled data, copied for in-place normalization
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
            # update position and scale
            self.pos[dim] = np.median(subdata)
            self.scale[dim] = subdata.std() or self.scale[dim] # never update scale to 0




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
    '''
    def keyPressEvent(self, event):
        print('keyPressEvent in cluster window')
        SpykeToolWindow.keyPressEvent(self, event) # pass it on
    '''
    def plot(self, X, nids):
        """Plot 3D projection of (possibly clustered) spike params in X"""
        X = X.copy() # make it contig
        self.glWidget.points = X
        self.glWidget.npoints = len(X)
        # don't like that I'm generating sids array from scratch, but there's no
        # such existing contiguous array in Sort or anywhere else that I can find
        self.glWidget.sids = np.arange(self.glWidget.npoints)
        self.glWidget.colors = CMAP[nids % len(CMAP)] # uint8
        self.glWidget.colors[nids == -1] = GREY # overwrite unclustered points with GREY
        self.glWidget.updateGL()
    '''
    def get_view(self):
        return mlab.view()

    def set_view(self, view):
        mlab.view(*view)

    view = property(get_view, set_view)

    def get_roll(self):
        return mlab.roll()

    def set_roll(self, roll):
        mlab.roll(roll)

    roll = property(get_roll, set_roll)
    '''

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.spw = self.parent().spykewindow
        #self.setMouseTracking(True) # req'd for tooltips purely on mouse motion, disabled for speed
        self.lastPos = QtCore.QPoint()
        self._dfocus = np.float32([0, 0, 0]) # accumulates changes to focus, in model coords

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

    def get_focus(self):
        return np.float32([0, 0, 0]) # focus is always at data origin, by definition

    def set_focus(self, xyz):
        """Set focus to x, y, z, in model coords"""
        xyz = np.float32(xyz)
        if not xyz.flags['OWNDATA']:
            # make sure modifying points in-place doesn't affect xyz arg
            xyz = xyz.copy()
        p = self.points
        #ipshell()
        t0 = time.time()
        # TODO: this works, but modifying vertex data is horribly inefficient and slow:
        p[:, 0] -= xyz[0]
        p[:, 1] -= xyz[1]
        p[:, 2] -= xyz[2]
        print('set_focus took %.3f sec' % (time.time()-t0))
        self._dfocus -= xyz # accumulate changes to focus

    focus = property(get_focus, set_focus)

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
        """From focus, or from data origin?"""
        v = self.getTranslation()
        return np.sqrt((v**2).sum()) # from data origin
        #return np.sqrt(((v-self.focus)**2).sum()) # from focus

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
        GL.glRotate(dangle, vr[0], vr[1], vr[2])

    def yaw(self, dangle): # aka azimuth
        """Rotate around view up vector"""
        vu = self.getViewUp()
        GL.glRotate(dangle, vu[0], vu[1], vu[2])

    def roll(self, dangle):
        """Rotate around view normal vector"""
        vn = self.getViewNormal()
        GL.glRotate(dangle, vn[0], vn[1], vn[2])

    def panTo(self, x, y):
        """Pan to absolute x and y position, leaving z unchanged"""
        MV = self.MV
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
        self.zoom(event.delta() / 1000)
        self.updateGL()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if modifiers == Qt.ControlModifier:
            if key == Qt.Key_Left:
                self.roll(5)
            elif key == Qt.Key_Right:
                self.roll(-5)
            elif key == Qt.Key_Up:
                self.zoom(0.05)
            elif key == Qt.Key_Down:
                self.zoom(-0.05)
        elif modifiers == Qt.ShiftModifier:
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
                self.focus = self._dfocus
                self.panTo(0, 0) # pan to new data origin
            elif key == Qt.Key_F: # reset focus to cursor position
                sid = self.pick(*self.cursorPosGL())
                if sid != None:
                    self.focus = self.points[sid]
                    self.panTo(0, 0) # pan to new data origin
            elif key == Qt.Key_S: # toggle item under the cursor, if any
                self.selectItemUnderCursor(clear=False)
            elif key == Qt.Key_Space: # clear and select item under cursor, if any
                self.selectItemUnderCursor(clear=True)
            elif key in [Qt.Key_Escape, Qt.Key_Delete, Qt.Key_D, Qt.Key_M, Qt.Key_NumberSign,
                         Qt.Key_O, Qt.Key_Period, Qt.Key_R]:
                sw = self.spw.windows['Sort']
                sw.keyPressEvent(event) # pass it on to Sort window
            elif key == Qt.Key_F11:
                self.parent().keyPressEvent(event) # pass it on to parent Cluster window

        self.updateGL()

    def showToolTip(self):
        """Pop up a nid or sid tooltip at current mouse cursor position"""
        #QtGui.QToolTip.hideText() # hide first if you want tooltip to move even when text is unchanged
        spw = self.spw
        sort = spw.sort
        x, y = self.cursorPosGL()
        sid = self.pick(x, y)
        if sid != None:
            dims = spw.GetClusterPlotDimNames()
            nid = sort.spikes[sid]['nid']
            tip = 'sid: %d\n' % sid
            sposstr = lst2shrtstr([ sort.spikes[sid][dim] for dim in dims ])
            tip += '%s: %s' % (lst2shrtstr(dims), sposstr)
            if nid > -1:
                tip += '\nnid: %d\n' % nid
                npoststr = lst2shrtstr([ sort.neurons[nid].cluster.pos[dim] for dim in dims ])
                tip += 'normed %s: %s' % (lst2shrtstr(dims), npoststr)
            globalPos = self.mapToGlobal(self.GLtoQt(x, y))
            QtGui.QToolTip.showText(globalPos, tip)
        else:
            QtGui.QToolTip.hideText()

    def selectItemUnderCursor(self, clear=False):
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
            spw.ToggleSpike(sid) # toggle its cluster too, if any
        self.showToolTip()
