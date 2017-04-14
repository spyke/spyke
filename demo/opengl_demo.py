"""Pared-down, free-standing version of GLWidget from spyke/cluster.py. For use as an OpenGL
widget demo. Most of the keyboard commands should work, including the arrow keys, S and D for
selection/deselection, F for focus, 0 for centering, and ? for a tooltip. You can zoom in and
out with the mouse wheel or the right button. CTRL rotates, SHIFT pans."""

from __future__ import division


import sys
import time
from copy import copy
import random
import numpy as np

from PyQt4 import QtCore, QtGui, QtOpenGL, uic
from PyQt4.QtCore import Qt
getSaveFileName = QtGui.QFileDialog.getSaveFileName
from OpenGL import GL, GLU

NPOINTS = 100000
VIEWDISTANCE = 2

# copied from spyke/core.py:
def iterable(x):
    """Check if the input is iterable, stolen from numpy.iterable()"""
    try:
        iter(x)
        return True
    except TypeError:
        return False

# copied from spyke/core.py:
def toiter(x):
    """Convert to iterable. If input is iterable, returns it. Otherwise returns it in a list.
    Useful when you want to iterate over something (like in a for loop),
    and you don't want to have to do type checking or handle exceptions
    when it isn't a sequence"""
    if iterable(x):
        return x
    else:
        return [x]

# copied from spyke/core.py:
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

# copied from spyke/core.py:
def tocontig(x):
    """Return C contiguous copy of array x if it isn't C contiguous already"""
    if not x.flags.c_contiguous:
        x = x.copy()
    return x

# colour stuff copied from spyke/plot.py:
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

DEFUVPERUM = 20
DEFUSPERUM = 100

BACKGROUNDCOLOUR = 'black'

PLOTCOLOURS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA,
               GREY, WHITE, BROWN]
CLUSTERCOLOURS = copy(PLOTCOLOURS)
CLUSTERCOLOURS.remove(GREY)

CLUSTERCOLOURSRGB = hex2rgb(CLUSTERCOLOURS)
GREYRGB = hex2rgb([GREY])[0] # pull it out of the list


class ClusterWindow(QtGui.QMainWindow):
    def __init__(self, pos=None, size=None):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle("opengl_demo.py")
        self.move(*pos)
        self.resize(*size)

        self.glWidget = GLWidget(parent=self)
        self.setCentralWidget(self.glWidget)

    def keyPressEvent(self, event):
        #key = event.key()
        #if key == Qt.Key_F11:
        #    SpykeToolWindow.keyPressEvent(self, event) # pass it up
        #else:
        self.glWidget.keyPressEvent(event) # pass it down
        
    def keyReleaseEvent(self, event):
        self.glWidget.keyReleaseEvent(event) # pass it down

    def plot(self, X, sids, nids):
        """Plot 3D projection of (possibly clustered) spike params in X"""
        X = tocontig(X) # ensure it's contig
        gw = self.glWidget
        gw.points = X
        gw.npoints = len(X)
        gw.sids = sids
        gw.nids = nids
        gw.colour() # set colours
        gw.updateGL()


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        #self.spw = self.parent().spykewindow
        #self.setMouseTracking(True) # req'd for tooltips purely on mouse motion, slow
        self.lastPressPos = QtCore.QPoint()
        self.lastPos = QtCore.QPoint()
        self.focus = np.float32([0, 0, 0]) # init camera focus
        self.axes = 'both' # display both mini and focal xyz axes by default
        self.selecting = None # True (selecting), False (deselecting), or None
        self.collected_sids = []
        #self.update_sigma()
        #self.spw.ui.sigmaSpinBox.valueChanged.connect(self.update_focal_axes)

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

    def colour(self, sids=None, sat=1):
        """Set colours of points corresponding to sids according to their nids, with
        saturation level sat. Caller is responsible for calling self.updateGL()"""
        if sids == None: # init/overwrite self.colours
            nids = self.nids
            # uint8, single unit nids are 1-based:
            self.colours = CLUSTERCOLOURSRGB[nids % len(CLUSTERCOLOURSRGB) - 1] * sat
            # overwrite unclustered/multiunit points with GREYRGB
            self.colours[nids < 1] = GREYRGB * sat
        else: # assume self.colours exists
            sidis = self.sids.searchsorted(sids)
            nids = self.nids[sidis]
            self.colours[sidis] = CLUSTERCOLOURSRGB[nids % len(CLUSTERCOLOURSRGB) - 1] * sat
            self.colours[sidis[nids < 1]] = GREYRGB * sat

    def initializeGL(self):
        # these are the defaults anyway, but just to be thorough:
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClearDepth(1.0)
        # display points according to occlusion, not order of plotting:
        GL.glEnable(GL.GL_DEPTH_TEST)
        # doesn't seem to work right, proper way to antialiase?:
        #GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glEnable(GL.GL_LINE_SMOOTH) # works better, makes lines thicker
        #GL.glPointSize(1.5) # truncs to the nearest pixel if antialiasing is off
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

        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glColorPointerub(self.colours) # should be n x rgb uint8, ie usigned byte
        GL.glVertexPointerf(self.points) # should be n x 3 contig float32
        GL.glDrawArrays(GL.GL_POINTS, 0, self.npoints)

        if self.axes: # paint xyz axes
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT) # make axes paint on top of data points
            if self.axes in ['both', 'mini']:
                self.paint_mini_axes()
            #if self.axes in ['both', 'focal']:
                #self.paint_focal_axes()

        # doesn't seem to be necessary, even though double-buffered mode is set with the
        # back buffer for RGB sid encoding. In fact, swapBuffers() call seems to cause
        # flickering, so leave disabled:
        #self.swapBuffers()

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        # fov (deg) controls amount of perspective, and as a side effect initial apparent size.
        # fov, aspect, nearz & farz clip planes:
        GLU.gluPerspective(45, width/height, 0.0001, 1000)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paint_mini_axes(self):
        """Paint mini xyz axes in bottom left of widget"""
        w, h = self.width(), self.height()
        vt = self.getTranslation() # this is in eye coordinates
        GL.glViewport(0, 0, w//8, h//8) # mini viewport at bottom left of widget
        self.setTranslation((0, 0, -3)) # draw in center of this mini viewport
        self.paint_axes()
        self.setTranslation(vt) # restore translation vector to MV matrix
        GL.glViewport(0, 0, w, h) # restore full viewport

    #def paint_focal_axes(self):
        #"""Paint xyz axes proportional in size to sigma, at focus"""
        #GL.glTranslate(*self.focus) # translate to focus
        #self.paint_axes(self.sigma)
        #GL.glTranslate(*-self.focus) # translate back

    def update_focal_axes(self):
        """Called every time sigma is changed in main spyke window"""
        #self.update_sigma()
        self.updateGL()

    #def update_sigma(self):
        #"""Update self.sigma from main spyke window"""
        #self.sigma = self.spw.ui.sigmaSpinBox.value()

    def paint_axes(self, l=1):
        """Paint axes at origin, with lines of length l"""
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(1, 0, 0) # red x axis
        GL.glVertex3f(0, 0, 0)
        GL.glVertex3f(l, 0, 0)
        GL.glColor3f(0, 1, 0) # green y axis
        GL.glVertex3f(0, 0, 0)
        GL.glVertex3f(0, l, 0)
        GL.glColor3f(0, 0, 1) # blue z axis
        GL.glVertex3f(0, 0, 0)
        GL.glVertex3f(0, 0, l)
        GL.glEnd()
    
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

    def lookDownXAxis(self):
        """Look down x axis: make x, y, z axes point out, right, and up"""
        MV = self.MV
        MV[:3, :3] = [[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]]
        self.MV = MV
        
    def lookUpXAxis(self):
        """Look up x axis: make x, y, z axes point in, left, and up"""
        MV = self.MV
        MV[:3, :3] = [[ 0, 0,-1],
                      [-1, 0, 0],
                      [ 0, 1, 0]]
        self.MV = MV
        
    def lookDownYAxis(self):
        """Look down y axis: make x, y, z axes point left, out, and up"""
        MV = self.MV
        MV[:3, :3] = [[-1, 0, 0],
                      [ 0, 0, 1],
                      [ 0, 1, 0]]
        self.MV = MV

    def lookUpYAxis(self):
        """Look up y axis: make x, y, z axes point right, in, and up"""
        MV = self.MV
        MV[:3, :3] = [[1, 0, 0],
                      [0, 0,-1],
                      [0, 1, 0]]
        self.MV = MV

    def lookDownZAxis(self):
        """Look down z axis: make x, y, z axes point down, right, and out"""
        MV = self.MV
        MV[:3, :3] = [[0,-1, 0],
                      [1, 0, 0],
                      [0, 0, 1]]
        self.MV = MV

    def lookUpZAxis(self):
        """Look up z axis: make x, y, z axes point up, right, and in"""
        MV = self.MV
        MV[:3, :3] = [[0, 1, 0],
                      [1, 0, 0],
                      [0, 0,-1]]
        self.MV = MV

    def rotateXOut(self):
        """Make x axis point out. Work on top left 3x3 subset of MV matrix.
        This was deduced by watching behaviour of MV matrix while manually
        rotating the x axis out. This is what we want, where a**2 + b**2 = 1:

        [0  0  1  *
         a -b  0  *
         b  a  0  *
         *  *  *  *]
        """
        MV = self.MV
        MV[:3, 2] = 1, 0, 0 # 3rd col is normal vector, make it point along x axis
        # set top left and top middle values to zero:
        MV[0, 0] = 0
        MV[0, 1] = 0
        b = MV[2, 0] # grab bottom left value
        a = np.sqrt(1 - b**2) # calc new complementary value to get normalized vectors
        #if MV[1, 0] < 0:
        #    a = -a # keep a -ve, reduce jumping around of axes
        MV[1, 0] = a
        MV[2, 1] = a
        MV[1, 1] = -b # needs to be -ve of MV[2, 0]
        self.MV = MV

    def rotateYRight(self):
        """Make y axis point right. Work on top left 3x3 subset of MV matrix.
        This was deduced by watching behaviour of MV matrix while manually
        rotating the y axis right. This is what we want, where a**2 + b**2 = 1:

        [0  a  b  *
         1  0  0  *
         0  b -a  *
         *  *  *  *]
        """
        MV = self.MV
        MV[:3, 0] = 0, 1, 0 # 1st col is right vector, make it point along y axis
        # set middle middle and middle right values to zero:
        MV[1, 1] = 0
        MV[1, 2] = 0
        a = MV[0, 1] # grab top middle value
        b = np.sqrt(1 - a**2) # calc new complementary value to get normalized vectors
        if MV[2, 1] < 0:
            b = -b # keep b -ve, reduce jumping around of axes
        MV[2, 1] = b
        MV[0, 2] = b
        MV[2, 2] = -a # needs to be -ve of MV[0, 1]
        self.MV = MV

    def rotateZUp(self):
        """Make z axis point up. Work on top left 3x3 subset of MV matrix.
        This was deduced by watching behaviour of MV matrix while manually
        rotating the z axis up. This is what we want, where a**2 + b**2 = 1:

        [a  0  b  *
         b  0 -a  *
         0  1  0  *
         *  *  *  *]
        """
        MV = self.MV
        MV[:3, 1] = 0, 0, 1 # 2nd col is up vector, make it point along z axis
        # set bottom left and bottom right z values to zero:
        MV[2, 0] = 0
        MV[2, 2] = 0
        a = MV[0, 0] # grab top left value
        b = np.sqrt(1 - a**2) # calc new complementary value to get normalized vectors
        if MV[1, 0] < 0:
            b = -b # keep b -ve, reduce jumping around of axes
        MV[1, 0] = b
        MV[0, 2] = b
        MV[1, 2] = -a # needs to be -ve of MV[0, 0]
        self.MV = MV

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

    def pick(self, x, y, pb=2, multiple=False):
        """Return sid of point at window coords x, y (bottom left origin),
        or first or multiple sids that fall within a square 2*pb+1 pix on a side,
        centered on x, y. pb is the pixel border to include around x, y"""
        width = self.size().width()
        height = self.size().height()
        #print('coords: %d, %d' % (x, y))
        # constrain to within border 1 pix smaller than widget, for glReadPixels call
        if not (pb <= x < width-pb and pb <= y < height-pb): # cursor out of range
            return
        if self.npoints > 2**24-2: # the last one is the full white background used as a no hit
            raise OverflowError("Can't pick from more than 2**24-2 sids")
        # draw encoded RGB values to back buffer
        #GL.glDrawBuffer(GL_BACK) # defaults to back
        GL.glClearColor(1.0, 1.0, 1.0, 1.0) # highest possible RGB means no hit
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glColorPointerub(self.rgbsids) # unsigned byte, ie uint8
        GL.glVertexPointerf(self.points) # float32
        GL.glDrawArrays(GL.GL_POINTS, 0, self.npoints) # to back buffer
        GL.glClearColor(0.0, 0.0, 0.0, 1.0) # restore to default black
        # grab back buffer:
        #GL.glReadBuffer(GL.GL_BACK) # defaults to back
        # find rgb at or around cursor coords, decode sid:
        backbuffer = GL.glReadPixels(x=x-pb, y=y-pb, width=2*pb+1, height=2*pb+1,
                                     format=GL.GL_RGB, type=GL.GL_UNSIGNED_BYTE,
                                     array=None, outputType=None)
        # NOTE: outputType kwarg above must be set to something other than str to ensure
        # that an array is returned, instead of a string of bytes
        if (backbuffer == 255).all(): # no hit
            return
        if not multiple:
            sid = self.decodeRGB(backbuffer[pb, pb]) # check center of backbuffer
            if sid != None:
                #print('hit at exact cursor pos')
                return sid # hit at exact cursor position
        # 2D array with nonzero entries at hits:
        hitpix = (backbuffer != [255, 255, 255]).sum(axis=2)
        if not multiple:
            ri = np.where(hitpix.ravel())[0][0] # get ravelled index of first hit
            i, j = np.unravel_index(ri, dims=hitpix.shape) # unravel to 2D index
            #print('hit at %d, %d' % (i, j))
            return self.decodeRGB(backbuffer[i, j]) # should be a valid sid
        ijs = zip(*np.where(hitpix)) # list of ij tuples
        return np.asarray([ self.decodeRGB(backbuffer[i, j]) for i, j in ijs ])

    def decodeRGB(self, rgb):
        """Convert encoded rgb value to sid"""
        r, g, b = rgb
        sid = r*65536 + g*256 + b
        if sid < 16777215: # 2**24 - 1
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
        """Record mouse position on button press, for use in mouseMoveEvent. On middle
        click, select spikes"""
        #sw = self.spw.windows['Sort']
        buttons = event.buttons()
        if buttons == QtCore.Qt.MiddleButton:
            #sw.on_actionSelectRandomSpikes_triggered()
            #sw.spykewindow.ui.plotButton.click() # same as hitting ENTER in nslist
            self.selecting = True
            self.setMouseTracking(True) # while selecting
            self.selectPointsUnderCursor()
        self.lastPressPos = QtCore.QPoint(event.pos())
        self.lastPos = QtCore.QPoint(event.pos())
    
    def mouseReleaseEvent(self, event):
        # seems have to use event.button(), not event.buttons(). I guess you can't
        # release multiple buttons simultaneously the way you can press them simultaneously?
        #sw = self.spw.windows['Sort']
        button = event.button()
        if button == QtCore.Qt.MiddleButton:
            if self.collected_sids:
                #self.spw.SelectSpikes(np.hstack(self.collected_sids), on=self.selecting)
                self.collected_sids = [] # clear it
            self.selecting = None
            self.setMouseTracking(False) # done selecting
        #elif button == QtCore.Qt.RightButton:
            #if QtCore.QPoint(event.pos()) == self.lastPressPos: # mouse didn't move
                #sw.on_actionSelectRandomSpikes_triggered()
    
    #def mouseDoubleClickEvent(self, event):
        #"""Clear selection, if any"""
        #if event.button() == QtCore.Qt.RightButton:
            #sw = self.spw.windows['Sort']
            #sw.clear()
    
    def mouseMoveEvent(self, event):
        buttons = event.buttons()

        if buttons != Qt.NoButton:
            modifiers = event.modifiers()
            shift = modifiers == Qt.ShiftModifier # only modifier is shift
            ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
            dx = event.x() - self.lastPos.x()
            dy = event.y() - self.lastPos.y()
            if buttons == QtCore.Qt.LeftButton:
                if shift:
                    self.pan(dx/700, -dy/700) # qt viewport y axis points down
                elif ctrl:
                    self.roll(-0.5*dx - 0.5*dy)
                else:
                    self.yaw(0.5*dx)
                    self.pitch(0.5*dy)
            elif buttons == QtCore.Qt.RightButton:
                #if shift or ctrl:
                #    self.spw.ui.sigmaSpinBox.stepBy(-dy)
                #else:
                self.zoom(-dy/500) # qt viewport y axis points down
            self.updateGL()
            self.lastPos = QtCore.QPoint(event.pos())

        if self.selecting != None:
            self.selectPointsUnderCursor()
        '''
        # pop up a tooltip on mouse movement, requires mouse tracking enabled
        if buttons == Qt.NoButton:
            self.showToolTip()
        else:
            QtGui.QToolTip.hideText()
        '''

    def wheelEvent(self, event):
        modifiers = event.modifiers()
        shift = Qt.ShiftModifier == modifiers # only modifier is shift
        ctrl = modifiers == Qt.ControlModifier # only modifier is ctrl
        #if shift or ctrl: # modify sigma
        #    # event.delta() seems to always be a multiple of 120 for some reason:
        #    self.spw.ui.sigmaSpinBox.stepBy(5 * event.delta() / 120)
        #else: # zoom
        self.zoom(event.delta() / 2000)
        self.updateGL()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        #sw = self.spw.windows['Sort']
        shift = Qt.ShiftModifier == modifiers # only modifier is shift
        ctrl = Qt.ControlModifier == modifiers # only modifier is ctrl
        if key == Qt.Key_Left:
            if shift:
                self.pan(-0.05, 0)
            elif ctrl:
                self.roll(5)
            else:
                self.yaw(-5)
        elif key == Qt.Key_Right:
            if shift:
                self.pan(0.05, 0)
            elif ctrl:
                self.roll(-5)
            else:
                self.yaw(5)
        elif key == Qt.Key_Up:
            if shift:
                self.pan(0, 0.05)
            elif ctrl:
                self.zoom(0.05)
            else:
                self.pitch(-5)
        elif key == Qt.Key_Down:
            if shift:
                self.pan(0, -0.05)
            elif ctrl:
                self.zoom(-0.05)
            else:
                self.pitch(5)
        elif key == Qt.Key_Question:
            self.showToolTip()
        elif key == Qt.Key_0: # reset focus to origin
            self.focus = np.float32([0, 0, 0])
            self.panTo() # pan to new focus
        elif key == Qt.Key_F: # reset focus to cursor position
            sid = self.pick(*self.cursorPosGL())
            if sid != None:
                self.focus = self.points[self.sids.searchsorted(sid)]
                self.panTo() # pan to new focus
        elif key == Qt.Key_A and ctrl: # cycle through xyz axes display, A on its own plots
            if self.axes == False:
                self.axes = 'both'
            elif self.axes == 'both':
                self.axes = 'mini'
            elif self.axes == 'mini':
                self.axes = 'focal'
            elif self.axes == 'focal':
                self.axes = False
        elif key == Qt.Key_1: # look along x axis
            if ctrl:
                self.lookUpXAxis()
            else:
                self.lookDownXAxis()
        elif key == Qt.Key_2: # look along y axis
            if ctrl:
                self.lookUpYAxis()
            else:
                self.lookDownYAxis()
        elif key == Qt.Key_3: # look along z axis
            if ctrl:
                self.lookUpZAxis()
            else:
                self.lookDownZAxis()
        elif key == Qt.Key_X: # make x axis point out
            self.rotateXOut()
        elif key == Qt.Key_Y: # make y axis point right
            self.rotateYRight()
        elif key == Qt.Key_Z: # make z axis point up
            self.rotateZUp()
        elif key == Qt.Key_S:
            if event.isAutoRepeat():
                return # event.ignore()?
            if shift:
                self.save()
            else: # select points under the cursor, if any
                self.selecting = True
                self.setMouseTracking(True) # while selecting
                self.selectPointsUnderCursor()
        elif key == Qt.Key_D: # deselect points under the cursor, if any
            if event.isAutoRepeat():
                return # event.ignore()?
            self.selecting = False
            self.setMouseTracking(True) # while deselecting
            self.selectPointsUnderCursor()
        elif key == Qt.Key_V: # V for View
            self.showProjectionDialog()            
        elif key in [Qt.Key_Enter, Qt.Key_Return]:
            pass #sw.spykewindow.ui.plotButton.click() # same as hitting ENTER in nslist
        #elif key == Qt.Key_F11:
        #    self.parent().keyPressEvent(event) # pass it on to parent Cluster window
        #elif key in [Qt.Key_A, Qt.Key_Escape, Qt.Key_Delete, Qt.Key_M, Qt.Key_G,
                     #Qt.Key_Equal, Qt.Key_Minus,
                     #Qt.Key_Slash, Qt.Key_P, Qt.Key_Backslash, Qt.Key_NumberSign, Qt.Key_R,
                     #Qt.Key_Space, Qt.Key_B, Qt.Key_Comma, Qt.Key_Period,
                     #Qt.Key_E, Qt.Key_C, Qt.Key_T, Qt.Key_W]:
            #sw.keyPressEvent(event) # pass it on to Sort window

        self.updateGL()

    def keyReleaseEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        shift = Qt.ShiftModifier == modifiers # only modifier is shift
        if not event.isAutoRepeat() and not shift and key in [Qt.Key_S, Qt.Key_D]:
            # stop selecting/deselecting
            if self.collected_sids:
                #self.spw.SelectSpikes(np.hstack(self.collected_sids), on=self.selecting)
                self.collected_sids = [] # clear it
            self.selecting = None
            self.setMouseTracking(False)

    def save(self):
        """Save cluster plot to file"""
        fname = getSaveFileName(self, "Save cluster plot to", 'cluster_plot.png')
        if fname:
            fname = str(fname) # convert from QString
            image = self.grabFrameBuffer() # defaults to withAlpha=False, makes no difference
            try:
                image.save(fname)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self.panel, "Error saving file", str(e),
                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)
            print('cluster plot saved to %r' % fname)


    def showToolTip(self):
        """Pop up a nid or sid tooltip at current mouse cursor position"""
        # hide first if you want tooltip to move even when text is unchanged:
        #QtGui.QToolTip.hideText()
        #spw = self.spw
        #sort = spw.sort
        x, y = self.cursorPosGL()
        sid = self.pick(x, y)
        if sid != None:
            #spos = []
            #dims = spw.GetClusterPlotDims()
            #for dim in dims:
                #if dim.startswith('c') and dim[-1].isdigit(): # it's a CA dim
                    #compid = int(lstrip(dim, 'c'))
                    #sidi = self.sids.searchsorted(sid)
                    #spos.append(sort.X[sort.Xhash][sidi, compid])
                #else: # it's a standard dim stored in spikes array
                    #spos.append(sort.spikes[sid][dim])
            tip = 'sid: %d' % sid
            #tip += '\n%s: %s' % (lst2shrtstr(dims), lst2shrtstr(spos))
            nid = self.nids[self.sids.searchsorted(sid)]#nid = sort.spikes[sid]['nid']
            if nid != 0:
                tip += '\nnid: %d' % nid
                #cpos = [ sort.neurons[nid].cluster.pos[dim] for dim in dims ]
                #tip += '\n%s: %s' % (lst2shrtstr(dims), lst2shrtstr(cpos))
            globalPos = self.mapToGlobal(self.GLtoQt(x, y))
            QtGui.QToolTip.showText(globalPos, tip)
        else:
            QtGui.QToolTip.hideText()

    def selectPointsUnderCursor(self):
        """Update point selection with those currently under cursor, within pixel border pb.
        Call this method on S and D down, and on mouse motion when either S or D are down"""
        #spw = self.spw
        #sw = spw.windows['Sort']
        #if clear:
        #    sw.uslist.clearSelection()
        #    sw.nlist.clearSelection()
        x, y = self.cursorPosGL()
        sids = self.pick(x, y, pb=10, multiple=True)
        if sids == None:
            return
        #t0 = time.time()
        #if not sw.panel.maxed_out:
        #    spw.SelectSpikes(sids, on=self.selecting)
        #else:
        #    # for speed, while the mouse is held down and the sort panel is maxed out,
        #    # don't call SelectSpikes, only call it once when the mouse is released
        self.collected_sids.append(sids)
        #print('SelectSpikes took %.3f sec' % (time.time()-t0))
        if self.selecting == True:
            sat = 0.2 # desaturate
        else: # self.selecting == False
            sat = 1 # resaturate
        self.colour(sids, sat=sat)
        self.updateGL()

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





if __name__ == '__main__':
    # prevents "The event loop is already running" messages when calling ipshell():
    QtCore.pyqtRemoveInputHook()
    app = QtGui.QApplication(sys.argv)
    cw = ClusterWindow(pos=(0, 0), size=(500, 500))
    cw.show()
    X = np.random.random((NPOINTS, 3)) - 0.5 # center them on 0
    X = np.float32(X)
    #import pdb; pdb.set_trace()
    sids = np.arange(NPOINTS)
    nids = sids % len(CLUSTERCOLOURSRGB)
    cw.plot(X, sids, nids)
    sys.exit(app.exec_())
