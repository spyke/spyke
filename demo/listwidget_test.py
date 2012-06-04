import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

NITEMS = 1000


class TestWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.list = TestListWidget(self)
        self.setCentralWidget(self.list)
        self.resize(500, 500)


class TestListView(QtGui.QListView):
    def __init__(self, parent):
        QtGui.QListView.__init__(self, parent)
        self.setModel(TestListModel(parent))
        self.setUniformItemSizes(True) # speeds up listview
        #self.setViewMode(QtGui.QListView.IconMode)
        #self.setFlow(QtGui.QListView.LeftToRight) # default is TopToBottom
        self.setWrapping(True) # default is False


class TestListModel(QtCore.QAbstractListModel):
    def __init__(self, parent):
        QtCore.QAbstractListModel.__init__(self, parent)

    def rowCount(self, parent=None):
        return NITEMS

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return index.row()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = TestWindow()
    win.show()
    try:
        from IPython import appstart_qt4
        appstart_qt4(app)
    except ImportError:
        sys.exit(app.exec_())
