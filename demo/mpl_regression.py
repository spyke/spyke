"""
regression: artist creation in custom Qt window 10X slower
line init speed regression in commit ca9d6b29595cf556504c457c8879fbc89c32a217 by mdboom on 2010-05-28. Compare to immediately preceding commit 7d7590e077125b4dcd12fdcf0f20d0091ccc97ba

3000 lines:
before: 0.9 sec
after: 11.9 sec
master: 11.9 sec

6000 lines:
before: 1.8 sec
after: 60 sec
master: 60 sec

"""
import sys
import time
from PyQt4 import QtGui

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

class MyFigureCanvasQTAgg(FigureCanvasQTAgg):
    def __init__(self):
        figure = Figure()
        FigureCanvasQTAgg.__init__(self, figure)
        self.ax = figure.add_axes([0, 0, 1, 1])
        t0 = time.time()
        self.init_lines()
        print('init_lines() took %.3f sec' % (time.time()-t0))

    def init_lines(self):
        for i in range(3000):
            line = Line2D([], [], visible=False)
            self.ax.add_line(line)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyFigureCanvasQTAgg()
    window.show()
    sys.exit(app.exec_())
