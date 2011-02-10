"""PyQt OpenGL example, modified from PySide example, which was a port of the opengl/hellogl
example from Qt v4.x"""
from __future__ import division

import sys
import math
from PyQt4 import QtCore, QtGui, QtOpenGL
from PyQt4.QtCore import Qt

from OpenGL import GL, GLU
import numpy as np

RED = 255, 0, 0
GREEN = 0, 255, 0
BLUE = 0, 0, 255
YELLOW = 255, 255, 0
MAGENTA = 255, 0, 255
CMAP = np.array([RED, GREEN, BLUE, YELLOW, MAGENTA], dtype=np.uint8)

def normdeg(angle):
    return angle % 360


class Window(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.glWidget = GLWidget(parent=self)
        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("OpenGL test"))

    def keyPressEvent(self, event):
        self.glWidget.keyPressEvent(event) # pass it down


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)

        #self.object = 0
        self.x = 0.0
        self.y = 0.0
        self.z = -2.0 # 0 would put us directly inside the random cube of points
        self.xrot = 0
        self.yrot = 0
        self.zrot = 0

        self.lastPos = QtCore.QPoint()
        '''
        format = QtOpenGL.QGLFormat()
        #format.setVersion(3, 0) # not available in PyQt 4.7.4
        # set to color index mode, unsupported in OpenGL >= 3.1, don't know how to load
        # GL_ARB_compatibility extension, and for now, can't force OpenGL 3.0 mode.
        # Gives "QGLContext::makeCurrent(): Cannot make invalid context current." error:
        format.setRgba(False)
        #format.setDoubleBuffer(True) # works fine
        self.setFormat(format)
        #QtOpenGL.QGLFormat.setDefaultFormat(format)

        c = QtGui.qRgb
        cmap = [c(255, 0, 0), c(0, 255, 0), c(0, 0, 255), c(255, 255, 0), c(255, 0, 255)]
        colormap = QtOpenGL.QGLColormap()
        colormap.setEntries(cmap)
        self.setColormap(colormap)
        '''
    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def initializeGL(self):
        # these are the defaults anyway, but just to be thorough
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClearDepth(1.0)

        #GL.glEnable(GL.GL_POINT_SMOOTH) # doesn't seem to work right, proper way to antialiase?
        #GL.glEnable(GL.GL_LINE_SMOOTH) # works better
        #GL.glPointSize(1.5) # truncs to the nearest pixel if antialiasing is off
        # float32 is much faster than float64
        self.npoints = 7000000
        self.points = np.float32(np.random.random((self.npoints, 3))) - 0.5
        self.cis = np.uint8(np.random.randint(5, size=self.npoints))
        self.colors = CMAP[self.cis] # uint8
        #n = int(self.npoints / 5)
        #self.n = n
        #self.i0 = np.arange(n, dtype=np.int32)
        #self.i1 = np.arange(n, 2*n, dtype=np.int32)
        #self.i2 = np.arange(2*n, 3*n, dtype=np.int32)
        #self.i3 = np.arange(3*n, 4*n, dtype=np.int32)
        #self.i4 = np.arange(4*n, 5*n, dtype=np.int32)


        GL.glEnable(GL.GL_DEPTH_TEST) # display points according to occlusion, not order of plotting

        #self.object = self.makeObject()
        #GL.glShadeModel(GL.GL_FLAT)
        #GL.glEnable(GL.GL_CULL_FACE) # only useful for solids

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # TODO: try storing vertex array in buffer object on server side (graphics memory?)
        # for faster rendering?

        # TODO: make rotation relative to eye coords, not modelview coords

        # this rotation and translation stuff seems wasteful here.
        # can't we just do this only on mouse/keyboard input, update the modelview
        # matrix, and then leave it as is in self.paintGL?
        GL.glLoadIdentity() # loads identity matrix into top of matrix stack
        GL.glTranslate(self.x, self.y, self.z) # zval zooms you in and out
        GL.glRotate(self.xrot, 1.0, 0.0, 0.0) # angles in deg
        GL.glRotate(self.yrot, 0.0, 1.0, 0.0)
        GL.glRotate(self.zrot, 0.0, 0.0, 1.0)

        #GL.glCallList(self.object)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY);
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
        #GL.glEnableClientState(GL.GL_INDEX_ARRAY);
        GL.glColorPointerub(self.colors) # usigned byte, ie uint8
        GL.glVertexPointerf(self.points) # float32
        #GL.glIndexPointeri(self.cis) # int32

        #GL.glColor(*RED)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.npoints)
        # might consider using buffer objects for even more speed (less unnecessary vertex
        # data from ram to vram, I think). Apparently, buffer objects don't work with
        # color arrays?

        #GL.glColor(*RED)
        #GL.glDrawArrays(GL.GL_POINTS, 0, self.npoints) # faster than glDrawElements?

        '''
        # glDrawArrays is a lot faster than glDrawElements:
        n = self.n
        GL.glColor(*RED)
        GL.glDrawArrays(GL.GL_POINTS, 0, n)
        GL.glColor(*GREEN)
        GL.glDrawArrays(GL.GL_POINTS, n, n)
        GL.glColor(*BLUE)
        GL.glDrawArrays(GL.GL_POINTS, 2*n, n)
        GL.glColor(*YELLOW)
        GL.glDrawArrays(GL.GL_POINTS, 3*n, n)
        GL.glColor(*MAGENTA)
        GL.glDrawArrays(GL.GL_POINTS, 4*n, n)
        '''
        '''
        GL.glColor(*RED)
        GL.glDrawElementsui(GL.GL_POINTS, self.i0) # treat array indices as unsigned int
        GL.glColor(*GREEN)
        GL.glDrawElementsui(GL.GL_POINTS, self.i1)
        GL.glColor(*BLUE)
        GL.glDrawElementsui(GL.GL_POINTS, self.i2)
        GL.glColor(*YELLOW)
        GL.glDrawElementsui(GL.GL_POINTS, self.i3)
        GL.glColor(*MAGENTA)
        GL.glDrawElementsui(GL.GL_POINTS, self.i4)
        '''
        #GL.glFlush() # forces drawing to begin, only makes difference for client-server?
        #self.swapBuffers()

    def resizeGL(self, width, height):
        #side = max(width, height)
        #GL.glViewport((width-side)/2, (height-side)/2, side, side)
        GL.glViewport(0, 0, width, height)
        # specify clipping box for perspective projection
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        #GL.glFrustum(-5, 5, -5, 5, 5, 20.) # nearz (2nd last arg) also zooms you in and out?
        # fov (deg) controls amount of perspective, and as a side effect initial apparent size
        GLU.gluPerspective(45, width/height, 0.0001, 1000) # fov, aspect, nearz & farz clip planes
        GL.glMatrixMode(GL.GL_MODELVIEW)

        # specify clipping box for orthonormal projection
        #GL.glMatrixMode(GL.GL_PROJECTION)
        #GL.glLoadIdentity()
        #GL.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        #GL.glMatrixMode(GL.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
        buttons = event.buttons()
        modifiers = event.modifiers()
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if buttons == QtCore.Qt.LeftButton:
            if modifiers == Qt.ControlModifier: # rotate around z
                self.zrot = normdeg(self.zrot - 0.5*dx - 0.5*dy) # rotates around the model's z axis, but really what we want is to rotate the scene around the viewer's normal axis
            elif modifiers == Qt.ShiftModifier: # translate in x and y
                self.x += dx / 100
                self.y -= dy / 100
            else: # rotate around x and y
                self.xrot = normdeg(self.xrot + 0.5*dy)
                self.yrot = normdeg(self.yrot + 0.5*dx)
        elif buttons == QtCore.Qt.RightButton: # zoom
            self.z -= dy / 40

        self.updateGL()
        self.lastPos = QtCore.QPoint(event.pos())

    def wheelEvent(self, event):
        self.z += event.delta() / 500 # zoom
        self.updateGL()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if modifiers == Qt.ControlModifier: # rotate around z, or zoom along z
            if key == Qt.Key_Left:
                self.zrot = normdeg(self.zrot + 5)
            elif key == Qt.Key_Right:
                self.zrot = normdeg(self.zrot - 5)
            elif key == Qt.Key_Up:
                self.z += 0.2
            elif key == Qt.Key_Down:
                self.z -= 0.2
        elif modifiers == Qt.ShiftModifier: # translate in x and y
            if key == Qt.Key_Left:
                self.x -= 0.2
            elif key == Qt.Key_Right:
                self.x += 0.2
            elif key == Qt.Key_Up:
                self.y += 0.2
            elif key == Qt.Key_Down:
                self.y -= 0.2
        else: # rotate around x and y
            if key == Qt.Key_Left:
                self.yrot = normdeg(self.yrot - 5)
            elif key == Qt.Key_Right:
                self.yrot = normdeg(self.yrot + 5)
            elif key == Qt.Key_Up:
                self.xrot = normdeg(self.xrot - 5)
            elif key == Qt.Key_Down:
                self.xrot = normdeg(self.xrot + 5)

        if key == Qt.Key_0: # reset focus
            self.x, self.y = 0.0, 0.0

        self.updateGL()

    '''
    # this specifies a display list, which is sent once, compiled, and then simply referenced
    # later every time the display needs to be updated. However, display lists are static once
    # compiled - none of their attributes can be changed
    def makeObject(self):
        genList = GL.glGenLists(1)
        GL.glNewList(genList, GL.GL_COMPILE)

        GL.glBegin(GL.GL_QUADS)

        x1 = +0.06
        y1 = -0.14
        x2 = +0.14
        y2 = -0.06
        x3 = +0.08
        y3 = +0.00
        x4 = +0.30
        y4 = +0.22

        self.quad(x1, y1, x2, y2, y2, x2, y1, x1)
        self.quad(x3, y3, x4, y4, y4, x4, y3, x3)

        self.extrude(x1, y1, x2, y2)
        self.extrude(x2, y2, y2, x2)
        self.extrude(y2, x2, y1, x1)
        self.extrude(y1, x1, x1, y1)
        self.extrude(x3, y3, x4, y4)
        self.extrude(x4, y4, y4, x4)
        self.extrude(y4, x4, y3, x3)

        Pi = 3.14159265358979323846
        NumSectors = 200

        for i in range(NumSectors):
            angle1 = (i * 2 * Pi) / NumSectors
            x5 = 0.30 * math.sin(angle1)
            y5 = 0.30 * math.cos(angle1)
            x6 = 0.20 * math.sin(angle1)
            y6 = 0.20 * math.cos(angle1)

            angle2 = ((i + 1) * 2 * Pi) / NumSectors
            x7 = 0.20 * math.sin(angle2)
            y7 = 0.20 * math.cos(angle2)
            x8 = 0.30 * math.sin(angle2)
            y8 = 0.30 * math.cos(angle2)

            self.quad(x5, y5, x6, y6, x7, y7, x8, y8)

            self.extrude(x6, y6, x7, y7)
            self.extrude(x8, y8, x5, y5)

        GL.glEnd()
        GL.glEndList()

        return genList

    def quad(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.qglColor(self.trolltechGreen)

        GL.glVertex3d(x1, y1, -0.05)
        GL.glVertex3d(x2, y2, -0.05)
        GL.glVertex3d(x3, y3, -0.05)
        GL.glVertex3d(x4, y4, -0.05)

        GL.glVertex3d(x4, y4, +0.05)
        GL.glVertex3d(x3, y3, +0.05)
        GL.glVertex3d(x2, y2, +0.05)
        GL.glVertex3d(x1, y1, +0.05)

    def extrude(self, x1, y1, x2, y2):
        self.qglColor(self.trolltechGreen.darker(250 + int(100 * x1)))

        GL.glVertex3d(x1, y1, +0.05)
        GL.glVertex3d(x2, y2, +0.05)
        GL.glVertex3d(x2, y2, -0.05)
        GL.glVertex3d(x1, y1, -0.05)
    '''
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
