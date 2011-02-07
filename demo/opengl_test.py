"""PyQt OpenGL example, modified from PySide example, which was a port of the opengl/hellogl
example from Qt v4.x"""
from __future__ import division


import sys
import math
from PyQt4 import QtCore, QtGui, QtOpenGL
from PyQt4.QtCore import Qt

from OpenGL import GL, GLU
import numpy as np


BLACK = 0., 0., 0., 1.
WHITE = 1., 1., 1.
RED = 1., 0., 0.
GREEN = 0., 1., 0.
BLUE = 0., 0., 1.
YELLOW = 1., 1., 0.
MAGENTA = 1., 0., 1.


def norm(angle):
    while angle < 0:
        angle += 360 * 16
    while angle > 360 * 16:
        angle -= 360 * 16
    return angle



class Window(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.glWidget = GLWidget()
        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("OpenGL test"))


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)

        #self.object = 0
        self.x = 0.0
        self.y = 0.0
        self.z = -2.0
        self.xrot = 0
        self.yrot = 0
        self.zrot = 0

        self.lastPos = QtCore.QPoint()
        #self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        #self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def initializeGL(self):
        GL.glClearColor(*BLACK) # seems to default to black anyway

        # TODO: maybe I should do something like this too:
        #GL.glClearDepth(1.0)

        #GL.glColor(*WHITE)
        #GL.glPointSize(2) # truncs to the nearest pixel
        # float32 is much faster than float64
        self.points = np.float32(np.random.random((100000, 3))) - 0.5
        npoints = len(self.points)/5
        self.i0 = np.arange(npoints, dtype=np.int32)
        self.i1 = np.arange(npoints, 2*npoints, dtype=np.int32)
        self.i2 = np.arange(2*npoints, 3*npoints, dtype=np.int32)
        self.i3 = np.arange(3*npoints, 4*npoints, dtype=np.int32)
        self.i4 = np.arange(4*npoints, 5*npoints, dtype=np.int32)

        #self.object = self.makeObject()
        #GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST) # displays points according to occlusion, not order of plotting
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
        #GL.glTranslated(0, 0, -13.)
        GL.glRotate(self.xrot / 16.0, 1.0, 0.0, 0.0) # anles in deg
        GL.glRotate(self.yrot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotate(self.zrot / 16.0, 0.0, 0.0, 1.0)

        #GL.glCallList(self.object)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
        #GL.glDrawArrays() sounds promising as an alternative
        GL.glVertexPointerf(self.points)
        GL.glColor(*RED)
        GL.glDrawElementsui(GL.GL_POINTS, self.i0)
        GL.glColor(*GREEN)
        GL.glDrawElementsui(GL.GL_POINTS, self.i1)
        GL.glColor(*BLUE)
        GL.glDrawElementsui(GL.GL_POINTS, self.i2)
        GL.glColor(*YELLOW)
        GL.glDrawElementsui(GL.GL_POINTS, self.i3)
        GL.glColor(*MAGENTA)
        GL.glDrawElementsui(GL.GL_POINTS, self.i4)

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
        GLU.gluPerspective(45, width/height, 0.0001, 50) # alternative
        GL.glMatrixMode(GL.GL_MODELVIEW)

        # specify clipping box for orthonormal projection
        #GL.glMatrixMode(GL.GL_PROJECTION)
        #GL.glLoadIdentity()
        #GL.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        #GL.glMatrixMode(GL.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
        modifiers = event.modifiers()
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() == QtCore.Qt.LeftButton:
            if modifiers == Qt.ControlModifier: # rotate around z
                self.zrot = norm(self.zrot - 8*dx - 8*dy) # rotates around the model's z axis, but really what we want is to rotate the scene around the viewer's normal axis
            elif modifiers == Qt.ShiftModifier: # translate in x and y
                self.x += dx / 100
                self.y -= dy / 100
            else: # rotate around x and y
                self.xrot = norm(self.xrot + 8*dy)
                self.yrot = norm(self.yrot + 8*dx)
        elif event.buttons() == QtCore.Qt.RightButton: # zoom
            self.z -= dy / 40

        self.updateGL()
        self.lastPos = QtCore.QPoint(event.pos())

    def wheelEvent(self, event):
        self.z += event.delta() / 500 # zoom
        self.updateGL()


    '''

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def xRotation(self):
        return self.xRot

    def yRotation(self):
        return self.yRot

    def zRotation(self):
        return self.zRot

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            #self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.updateGL()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            #self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.updateGL()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            #self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
            self.updateGL()
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
